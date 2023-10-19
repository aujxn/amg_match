/*
use std::borrow::Borrow;

use amg_match::{
    adaptive::build_adaptive,
    partitioner::modularity_matching,
    preconditioner::{
        BackwardGaussSeidel as Bgs, ForwardGaussSeidel as Fgs, Multilevel, PcgL1, Preconditioner,
        SymmetricGaussSeidel as Sgs, L1,
    },
    solver::{pcg, stationary},
    utils::{delete_boundary, load_boundary_dofs, load_vec, random_vec},
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
    /*
    pretty_env_logger::init();
    let opt = Opt::from_args();

    let mat = CsrMatrix::from(
        &nalgebra_sparse::io::load_coo_from_matrix_market_file(&opt.input).unwrap(),
    );

    let b = load_vec("data/spe10/spe10_0.rhs");
    let dofs = load_boundary_dofs("data/spe10/spe10_0.bdy");

    let (mat, b) = delete_boundary(dofs, mat, b);

    /*
    let norm = mat
        .values()
        .iter()
        .fold(0.0_f64, |acc, x| acc + x * x)
        .sqrt();
    mat = norm;
    */
    /*
    let mut diag: CsrMatrix<f64> = mat.diagonal_as_csr();
    diag.triplet_iter_mut()
        .for_each(|(_, _, val)| *val = 1.0 / val.sqrt());
    let mat = &diag * &mat * &diag;
    */

    let dim = mat.nrows();
    let ones = DVector::from(vec![1.0; mat.nrows()]);
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut x = random_vec(dim);

    let test: DVector<f64> = &mat * &ones;
    let zero_counter = test.into_iter().filter(|val| (**val).abs() < 1e-6).count();
    if zero_counter > 0 {
        warn!("zero rows in mat: {}", zero_counter);
    }
    let zero_counter = b.into_iter().filter(|val| (**val).abs() < 1e-6).count();
    if zero_counter > 0 {
        warn!("zero rows in rhs: {}", zero_counter);
    }

    let mat = std::rc::Rc::new(mat);
    let timer = std::time::Instant::now();
    let preconditioner: Box<dyn Preconditioner> = match opt.preconditioner {
        PreconditionerArg::L1 => Box::new(L1::new(&mat)),
        PreconditionerArg::Fgs => Box::new(Fgs::new(&mat)),
        PreconditionerArg::Bgs => Box::new(Bgs::new(&mat)),
        PreconditionerArg::Sgs => Box::new(Sgs::new(&mat)),
        PreconditionerArg::Adaptive => {
            let pc = Box::new(build_adaptive(mat.clone(), 3.0, 10));
            pc.save("data/out/test_pc.json", "first test".into());
            pc
        }
        _ => {
            let iterations_for_near_null = 10;
            info!(
                "calculating near null component... {} iterations using stationary L1",
                iterations_for_near_null
            );

            let bmat: &CsrMatrix<f64> = mat.borrow();
            let test: DVector<f64> = bmat * &ones;
            warn!("{:?}", test.norm());

            let mut x: DVector<f64> = random_vec(dim);
            let _converged = pcg(
                &mat,
                &zeros,
                &mut x,
                iterations_for_near_null,
                1e-16,
                &mut L1::new(&mat),
                None,
            );

            //x /= x.norm();
            x.normalize_mut();
            let hierarchy = modularity_matching(mat.clone(), &x, 4.0);
            info!(
                "Number of levels in hierarchy: {}",
                hierarchy.get_matrices().len(),
            );
            info!(
                "Size of coarsest: {}",
                hierarchy.get_matrices().last().unwrap().nrows()
            );

            match opt.preconditioner {
                PreconditionerArg::Ml1 => Box::new(Multilevel::<PcgL1>::new(hierarchy)),
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
            &zeros,
            &mut x,
            opt.max_iter,
            opt.tolerance,
            &*preconditioner,
            Some(3),
        ),
        Solver::Pcg => {
            pcg(
                &mat,
                &b,
                &mut x,
                opt.max_iter,
                opt.tolerance,
                &*preconditioner,
                Some(3),
            )
            .0
        }
    };
    info!("Solved in: {} ms.", timer.elapsed().as_millis());
    */
}
*/
fn main() {}
