pub mod adaptive;
pub mod parallel_ops;
pub mod partitioner;
pub mod preconditioner;
pub mod solver;
pub mod suitesparse_dl;

#[macro_use]
extern crate log;

pub fn random_vec(size: usize) -> nalgebra::DVector<f64> {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::new(-2.0_f64, 2.0_f64);
    nalgebra::DVector::from_distribution(size, &distribution, &mut rng)
}
