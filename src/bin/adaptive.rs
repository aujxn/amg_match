/*
use amg_match::{
    adaptive::adaptive,
    //mat_to_image,
    partitioner::{modularity_matching, modularity_matching_no_copies},
    preconditioner::{bgs, fgs, l1, multilevelgs, multilevell1, sgs},
    solver::{pcg, stationary},
};
*/

use amg_match::preconditioner_new::{
    BackwardGaussSeidel as Bgs, ForwardGaussSeidel as Fgs, SymmetricGaussSeidel as Sgs, L1,
};
use amg_match::solver::stationary;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use rand::{distributions::Uniform, thread_rng};
use structopt::StructOpt;
use strum_macros::{Display, EnumString};

#[macro_use]
extern crate log;

//TODO make preconditioner construction and impl parallel
//     and replace decomp with LAPACK

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
    preconditioner: Preconditioner,

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
enum Preconditioner {
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

    let dim = mat.nrows();
    let ones = DVector::from(vec![1.0; mat.nrows()]);
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let x: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
    let b = &mat * &x;

    let preconditioner: Box<dyn amg_match::preconditioner_new::Preconditioner> =
        match opt.preconditioner {
            Preconditioner::L1 => Box::new(L1::new(&mat)),
            Preconditioner::Fgs => Box::new(Fgs::new(&mat)),
            Preconditioner::Bgs => Box::new(Bgs::new(&mat)),
            Preconditioner::Sgs => Box::new(Sgs::new(&mat)),
            _ => unimplemented!(),
            /*
                Preconditioner::Adaptive => adaptive(&mat),
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

                    let (near_null, _) = stationary(
                        &mat,
                        &zeros,
                        &x,
                        iterations_for_near_null,
                        10.0_f64.powi(-6),
                        &l1(&mat),
                    );

                    let hierarchy = modularity_matching(mat.clone(), &near_null, 2.0);
                    info!(
                        "Number of levels in hierarchy: {}",
                        hierarchy.get_matrices().len(),
                    );
                    info!(
                        "Size of coarsest: {}",
                        hierarchy.get_matrices().last().unwrap().nrows()
                    );

                    match opt.preconditioner {
                        Preconditioner::Ml1 => multilevell1(hierarchy),
                        Preconditioner::Mgs => multilevelgs(hierarchy),
                        _ => unimplemented!(),
                    }
                }
            */
        };

    info!("solving");
    let _rhs = match opt.solver {
        Solver::Stationary => stationary(
            &mat,
            &b,
            &zeros,
            opt.max_iter,
            opt.tolerance,
            &*preconditioner,
        ),
        _ => unimplemented!(),
        /*
        Solver::Pcg => pcg(
            &mat,
            &b,
            &zeros,
            opt.max_iter,
            opt.tolerance,
            &preconditioner,
        ),
        */
    };
}
