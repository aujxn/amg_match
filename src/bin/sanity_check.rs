use std::{sync::Arc, time::Duration};

use amg_match::{
    adaptive::AdaptiveBuilder,
    interpolation::InterpolationType,
    preconditioner::{BlockSmootherType, SmootherType},
    solver::{Iterative, IterativeMethod, LogInterval},
    utils::load_system,
    Vector,
};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

#[macro_use]
extern crate log;

fn main() {
    pretty_env_logger::init();

    let epsilon = 1e-12;
    let mut results: Vec<(usize, usize, usize)> = Vec::new();
    //let mut results: Vec<(usize, usize)> = Vec::new();
    //let method = IterativeMethod::StationaryIteration;
    let method = IterativeMethod::ConjugateGradient;

    let smoother = SmootherType::DiagonalCompensatedBlock(
        //BlockSmootherType::AutoCholesky(sprs::FillInReduction::CAMDSuiteSparse),
        //BlockSmootherType::GaussSeidel,
        BlockSmootherType::IncompleteCholesky,
        64,
    );
    let interpolator = InterpolationType::SmoothedAggregation((1, 0.66));

    for i in 2..8 {
        let prefix = "data/laplace";
        let name = format!("{}", i);
        let (mat, b, _coords, _rbms, _projector) = load_system(&prefix, &name, false);
        let dim = mat.rows();
        //let cf = dim as f64 / 100.0;
        let cf = 8.0;
        let max_iter = 5000;
        info!("Starting: {}", i);

        //let rand: Vector = Vector::from_element(b.len(), 1.0).normalize();
        //let b: Vector = random_vec(dim);

        let adaptive_builder = AdaptiveBuilder::new(mat.clone())
            .with_max_components(1)
            .with_smoother(smoother)
            .with_interpolator(interpolator)
            .with_smoothing_steps(1)
            .with_max_test_iters(10)
            .with_coarsening_factor(cf);

        let (multi_pc, _, _) = adaptive_builder.build();
        let multi_pc = Arc::new(multi_pc);

        let adaptive_builder = adaptive_builder.with_max_level(2);
        let (two_pc, _, _) = adaptive_builder.build();
        let two_pc = Arc::new(two_pc);

        let avg = 1;

        let mut multi_sum = 0;
        let mut two_sum = 0;

        for _ in 0..avg {
            let x = Vector::random(dim, Uniform::new(-1., 1.));
            multi_pc.components()[0].get_hierarchy().print_table();
            let multi_solver = Iterative::new(mat.clone(), Some(x))
                .with_relative_tolerance(epsilon)
                .with_max_iter(max_iter)
                .with_solver(method)
                .with_preconditioner(multi_pc.clone())
                .with_log_interval(LogInterval::Time(Duration::from_secs(30)));
            let (_, multi_solve_result) = multi_solver.solve(&b);

            if !multi_solve_result.converged {
                warn!(
                    "multi level didn't converge with ratio: {}",
                    multi_solve_result.final_relative_residual_norm
                );
            }
            multi_sum += multi_solve_result.iterations;

            let x = Vector::random(dim, Uniform::new(-1., 1.));
            let two_solver = Iterative::new(mat.clone(), Some(x))
                .with_relative_tolerance(epsilon)
                .with_max_iter(max_iter)
                .with_solver(method)
                .with_preconditioner(two_pc.clone())
                .with_log_interval(LogInterval::Time(Duration::from_secs(30)));
            let (_, two_solve_result) = two_solver.solve(&b);

            if !two_solve_result.converged {
                warn!(
                    "two level didn't converge with ratio: {}",
                    two_solve_result.final_relative_residual_norm
                );
            }
            two_sum += two_solve_result.iterations;
        }

        results.push((dim, two_sum / avg, multi_sum / avg));
        //results.push((dim, multi_sum / avg));
        //results.push((dim, two_sum, 0));

        println!("{:>15} {:>15} {:>15}", "size", "2-level", "multi-level");
        for (dim, two, multi) in results.iter() {
            println!("{:15} {:15} {:15}", dim, two, multi);
        }
        /*
        println!("{:>15} {:>15}", "size", "multi-level");
        for (dim, multi) in results.iter() {
            println!("{:15} {:15}", dim, multi);
        }
        */
    }
}
