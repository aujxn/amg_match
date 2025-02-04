use std::{sync::Arc, time::Duration};

use amg_match::{
    adaptive::AdaptiveBuilder,
    partitioner::InterpolationType,
    solver::{Iterative, IterativeMethod, LogInterval},
    utils::{load_system, random_vec},
};

#[macro_use]
extern crate log;

fn main() {
    pretty_env_logger::init();

    let epsilon = 1e-12;
    let mut results: Vec<(usize, usize, usize)> = Vec::new();
    //let mut results: Vec<(usize, usize)> = Vec::new();
    let method = IterativeMethod::StationaryIteration;

    for i in 2..8 {
        let prefix = format!("data/laplace/{}", i);
        let (mat, b, _coords, _projector) = load_system(&prefix);
        let dim = mat.rows();
        //let cf = dim as f64 / 100.0;
        let cf = 8.0;
        let max_iter = 5000;
        info!("Starting: {}", i);

        //let rand: Vector = Vector::from_element(b.len(), 1.0).normalize();
        //let b: Vector = random_vec(dim);

        let adaptive_builder = AdaptiveBuilder::new(mat.clone())
            .with_max_components(1)
            .with_smoother(amg_match::preconditioner::SmootherType::L1)
            //.with_smoother(SmootherType::BlockL1)
            .with_interpolator(InterpolationType::SmoothedAggregation(1))
            //.with_interpolator(InterpolationType::UnsmoothedAggregation)
            .with_smoothing_steps(3)
            .with_max_test_iters(50)
            .with_coarsening_factor(cf);
        let (multi_pc, _, _) = adaptive_builder.build();
        let multi_pc = Arc::new(multi_pc);

        let adaptive_builder = AdaptiveBuilder::new(mat.clone())
            .with_max_components(1)
            .with_max_level(2)
            .with_smoother(amg_match::preconditioner::SmootherType::L1)
            //.with_smoother(SmootherType::BlockL1)
            .with_interpolator(InterpolationType::SmoothedAggregation(1))
            //.with_interpolator(InterpolationType::UnsmoothedAggregation)
            .with_smoothing_steps(3)
            .with_max_test_iters(50)
            .with_coarsening_factor(cf);
        let (two_pc, _, _) = adaptive_builder.build();
        let two_pc = Arc::new(two_pc);

        let avg = 1;

        let mut multi_sum = 0;
        let mut two_sum = 0;

        for _ in 0..avg {
            let x: Vector = random_vec(dim);
            let multi_solver = Iterative::new(mat.clone(), Some(x))
                .with_tolerance(epsilon)
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

            let x: Vector = random_vec(dim);
            let two_solver = Iterative::new(mat.clone(), Some(x))
                .with_tolerance(epsilon)
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
