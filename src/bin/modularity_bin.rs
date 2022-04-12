use amg_match::partitioner::modularity_matching;
use amg_match::preconditioner::{l1, multilevel};
use amg_match::solver::{pcg, stationary};
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::path::PathBuf;
use structopt::StructOpt;

#[macro_use]
extern crate log;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Matrix file in matrix market format
    #[structopt(parse(from_os_str))]
    input: PathBuf,
    //TODO: add cli for rhs option?
}

fn main() {
    pretty_env_logger::init();
    let opt = Opt::from_args();

    let mat = sprs::io::read_matrix_market::<f64, usize, _>(&opt.input)
        .unwrap()
        .to_csr::<usize>();
    let dim = mat.rows();
    let ones = ndarray::Array::from_vec(vec![1.0; mat.rows()]);
    let zeros = ndarray::Array::from_vec(vec![0.0; mat.rows()]);
    let x: Array1<f64> = ndarray::Array::random(dim, Uniform::new(-2., 2.));
    let b = &mat * &x;

    info!("Stationary iterative method with L1");
    let l1_precond = l1(&mat);
    let _ = stationary(&mat, &b, &zeros, 10000, 10.0_f64.powi(-6), &l1_precond);

    info!("PCG with L1 as preconditioner");
    let l1_precond = l1(&mat);
    let _ = pcg(&mat, &b, &zeros, 10000, 10.0_f64.powi(-6), &l1_precond);

    info!("Begining construction of multilevel preconditioner");
    let l1_precond = l1(&mat);

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
        &l1_precond,
    );

    info!("Building partition hierarchy...");
    let hierarchy = modularity_matching(mat.clone(), &near_null, 2.0);
    info!(
        "Number of levels in hierarchy: {}",
        hierarchy.get_matrices().len()
    );
    info!("Building multilevel preconditioner...");
    let multilevel_preconditioner = multilevel(hierarchy);

    info!("Performing symmetry check on multilevel preconditioner...");
    for _ in 0..50 {
        let u = ndarray::Array::random(dim, Uniform::new(0., 2.));
        let v = ndarray::Array::random(dim, Uniform::new(0., 2.));
        let left: f64 = u.t().dot(&v);
        let right: f64 = v.t().dot(&u);
        assert!(left - right < 10.0_f64.powi(-6));
        let pos: f64 = u.t().dot(&u);
        assert!(pos > 0.0);
        let pos: f64 = v.t().dot(&v);
        assert!(pos > 0.0);
    }
    info!("symmetry OK");

    info!("Stationary iterative method with multilevel L1");
    let _x = stationary(
        &mat,
        &b,
        &zeros,
        10000,
        10.0_f64.powi(-6),
        &multilevel_preconditioner,
    );

    info!("PCG with multilevel as preconditioner");
    let _x = pcg(
        &mat,
        &b,
        &zeros,
        10000,
        10.0_f64.powi(-6),
        &multilevel_preconditioner,
    );
}
