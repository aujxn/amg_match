use amg_match::{
    adaptive::build_adaptive,
    //mat_to_image,
    partitioner::modularity_matching,
    random_vec,
    //preconditioner::{bgs, fgs, l1, multilevelgs, multilevell1, sgs},
    solver::{pcg, stationary},
};

use amg_match::preconditioner::{
    BackwardGaussSeidel as Bgs, ForwardGaussSeidel as Fgs, Multilevel, Preconditioner,
    SymmetricGaussSeidel as Sgs, L1,
};
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use structopt::StructOpt;
use strum_macros::{Display, EnumString};

#[macro_use]
extern crate log;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "amg_match_cli",
    about = "CLI for testing different iterative solvers and preconditioners"
)]
struct Opt {
    /// Matrix file in matrix market format
    #[structopt(parse(from_os_str))]
    input: std::path::PathBuf,

    /// Maximum number of iterations to perform
    max_iter: usize,

    /// Method to use with the solver. Options are:
    /// l1, sgs, fgs, bgs, ml1, mgs
    preconditioner: PreconditionerArg,

    /// Solver to use. Options are:
    /// pcg, stationary
    solver: Solver,

    /// Stop iterations after scaled residual is less
    /// than tolerance squared
    #[structopt(default_value = "1e-6")]
    tolerance: f64,
    //#[structopt(short, long)]
    //picture: Option<String>,
    //TODO: add cli for rhs option?
}

#[derive(Debug, Display, EnumString)]
#[strum(ascii_case_insensitive)]
enum PreconditionerArg {
    L1,
    Fgs,
    Bgs,
    Sgs,
    Ml1,
    Mgs,
    Adaptive,
}

#[derive(Debug, Display, EnumString)]
#[strum(ascii_case_insensitive)]
enum Solver {
    Pcg,
    Stationary,
}

fn main() {
    pretty_env_logger::init();
    let opt = Opt::from_args();

    let mat = CsrMatrix::from(
        &nalgebra_sparse::io::load_coo_from_matrix_market_file(&opt.input).unwrap(),
    );

    /*
    let norm = mat
        .values()
        .iter()
        .fold(0.0_f64, |acc, x| acc + x * x)
        .sqrt();
    mat = norm;
    */
    let mut diag: CsrMatrix<f64> = mat.diagonal_as_csr();
    diag.triplet_iter_mut()
        .for_each(|(_, _, val)| *val = 1.0 / val.sqrt());
    let mat = &diag * &mat * &diag;

    let dim = mat.nrows();
    let _ones = DVector::from(vec![1.0; mat.nrows()]);
    let mut zeros = DVector::from(vec![0.0; mat.nrows()]);
    let x: DVector<f64> = random_vec(dim);
    let b = &mat * &x;

    let timer = std::time::Instant::now();
    let mut preconditioner: Box<dyn Preconditioner> = match opt.preconditioner {
        PreconditionerArg::L1 => Box::new(L1::new(&mat)),
        PreconditionerArg::Fgs => Box::new(Fgs::new(&mat)),
        PreconditionerArg::Bgs => Box::new(Bgs::new(&mat)),
        PreconditionerArg::Sgs => Box::new(Sgs::new(&mat)),
        PreconditionerArg::Adaptive => Box::new(build_adaptive(&mat)),
        _ => {
            let iterations_for_near_null = 10;
            info!(
                "calculating near null component... {} iterations using stationary L1",
                iterations_for_near_null
            );

            /*
            let test = &mat * &ones;
            info!("{:?}", test);
            */

            let mut x: DVector<f64> = random_vec(dim);
            let _converged = stationary(
                &mat,
                &zeros,
                &mut x,
                iterations_for_near_null,
                10.0_f64.powi(-6),
                &mut L1::new(&mat),
                None,
            );

            let hierarchy = modularity_matching(&mat, &x, 2.0);
            info!(
                "Number of levels in hierarchy: {}",
                hierarchy.get_matrices().len(),
            );
            info!(
                "Size of coarsest: {}",
                hierarchy.get_matrices().last().unwrap().nrows()
            );

            match opt.preconditioner {
                PreconditionerArg::Ml1 => Box::new(Multilevel::<L1>::new(hierarchy)),
                PreconditionerArg::Mgs => unimplemented!(),
                _ => unreachable!(),
            }
        }
    };

    info!(
        "Preconitioner built in: {} seconds. solving...",
        timer.elapsed().as_secs()
    );
    let timer = std::time::Instant::now();
    let _rhs = match opt.solver {
        Solver::Stationary => stationary(
            &mat,
            &b,
            &mut zeros,
            opt.max_iter,
            opt.tolerance,
            &mut *preconditioner,
            Some(50),
        ),
        Solver::Pcg => pcg(
            &mat,
            &b,
            &mut zeros,
            opt.max_iter,
            opt.tolerance,
            &mut *preconditioner,
            Some(1),
        ),
    };
    info!("Solved in: {} ms.", timer.elapsed().as_millis());
}
