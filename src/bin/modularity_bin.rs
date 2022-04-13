use amg_match::{
    //mat_to_image,
    partitioner::modularity_matching,
    preconditioner::{bgs, fgs, l1, multilevel, sgs},
    solver::{pcg, stationary},
};
use ndarray::Array1;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
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
    preconditioner: Preconditioner,

    /// Solver to use. Options are:
    /// pcg, stationary
    solver: Solver,

    /// Stop iterations after scaled residual is less
    /// than tolerance squared
    #[structopt(default_value = "10e-6")]
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

    let mat = sprs::io::read_matrix_market::<f64, usize, _>(&opt.input)
        .unwrap()
        .to_csr::<usize>();
    /*
    if let Some(file_out) = opt.picture {
        mat_to_image(&mat, &file_out);
    }
    */

    let dim = mat.rows();
    let ones = ndarray::Array::from_vec(vec![1.0; mat.rows()]);
    let zeros = ndarray::Array::from_vec(vec![0.0; mat.rows()]);
    let x: Array1<f64> = ndarray::Array::random(dim, Uniform::new(-2., 2.));
    let b = &mat * &x;

    let preconditioner = match opt.preconditioner {
        Preconditioner::L1 => l1(&mat),
        Preconditioner::Fgs => fgs(&mat),
        Preconditioner::Bgs => bgs(&mat),
        Preconditioner::Sgs => sgs(&mat),
        Preconditioner::Ml1 => {
            let iterations_for_near_null = 5;
            info!(
                "calculating near null component... {} iterations using stationary L1",
                iterations_for_near_null
            );
            let (near_null, _) = stationary(
                &mat,
                &zeros,
                &ones,
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
                hierarchy.get_matrices().last().unwrap().rows()
            );

            /* TODO: write test to do this
            for p in hierarchy.get_partitions().iter() {
                let ones = ndarray::Array::from_vec(vec![1.0; p.cols()]);
                let result = p * &ones;
                info!("{:?}", result);
            }
            */

            multilevel(hierarchy)
        }
        Preconditioner::Mgs => unimplemented!(),
    };

    /* TODO: this is not correct but add test for precond symmetry
    for _ in 0..50 {
        let mut u = ndarray::Array::random(dim, Uniform::new(0., 2.));
        let mut v = ndarray::Array::random(dim, Uniform::new(0., 2.));
        let left: f64 = u.t().dot(&v);
        let right: f64 = v.t().dot(&u);
        assert!(left - right < 10.0_f64.powi(-6));
        let pos: f64 = u.t().dot(&u);
        assert!(pos > 0.0);
        let pos: f64 = v.t().dot(&v);
        assert!(pos > 0.0);
    }
    */

    let _rhs = match opt.solver {
        Solver::Stationary => stationary(
            &mat,
            &b,
            &zeros,
            opt.max_iter,
            opt.tolerance,
            &preconditioner,
        ),
        Solver::Pcg => pcg(
            &mat,
            &b,
            &zeros,
            opt.max_iter,
            opt.tolerance,
            &preconditioner,
        ),
    };
}
