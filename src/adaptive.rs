//! This module contains the code to construct the adaptive preconditioner.

use crate::{
    utils::{normalize, random_vec},
    partitioner::{modularity_matching, modularity_matching_add_level, Hierarchy},
    preconditioner::{Composite, CompositeType, Multilevel, PcgL1, Preconditioner, L1},
    solver::pcg
};
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use plotly::{
    layout::{Axis, AxisType, Layout},
    Plot, Scatter,
};
use rand::prelude::*;
use std::time::{Duration, Instant};

pub fn build_adaptive<'a>(mat: &'a CsrMatrix<f64>) -> Composite<'a> {
    let dim = mat.nrows();
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut near_null = random_vec(dim);
    let mut preconditioner = Composite::new(mat, Vec::new(), CompositeType::Sequential);
    let mut l1 = L1::new(mat);
    let _ = crate::solver::stationary(mat, &zeros, &mut near_null, 10, 1e-16, &mut l1, None);
    //near_null /= near_null.norm();
    //near_null.normalize_mut();
    //no_zeroes(&mut near_null);
    normalize(&mut near_null, mat);
    let coarsening_factor = 4.0;
    let hierarchy = modularity_matching(&mat, &near_null, coarsening_factor);
    let ml1 = Multilevel::<PcgL1>::new(hierarchy);
    preconditioner.push(Box::new(ml1));

    loop {
        near_null = random_vec(dim);
        let _ = pcg(mat, &zeros, &mut near_null, 10, 1e-16, &mut l1, None);

        if let Some((convergence_rate, _)) =
            find_near_null(mat, &mut preconditioner, &mut near_null)
        {
            //near_null /= near_null.norm();
            //near_null.normalize_mut();
            //no_zeroes(&mut near_null);
            normalize(&mut near_null, mat);

            let hierarchy = modularity_matching(&mat, &near_null, coarsening_factor);
            let ml1 = Multilevel::<PcgL1>::new(hierarchy);
            preconditioner.push(Box::new(ml1));
            if convergence_rate < 0.6 || preconditioner.components().len() == 3 {
                return preconditioner;
            }
        } else {
            return preconditioner;
        }
    }
}

pub fn build_adaptive_new<'a>(mat: &'a CsrMatrix<f64>, coarsening_factor: f64) -> Composite<'a> {
    let mut preconditioner = Composite::new(mat, Vec::new(), CompositeType::Sequential);

    let mut plot = Plot::new();
    let dim = mat.nrows();
    let mut l1 = L1::new(mat);
    let zeros = DVector::from(vec![0.0; dim]);
    let mut near_null = random_vec(dim);
    let _ = crate::solver::stationary(mat, &zeros, &mut near_null, 50, 1e-16, &mut l1, None);
    let mut near_null_history = Vec::<DVector<f64>>::new();

    loop {
        let mut hierarchy = Hierarchy::new(mat);
        //near_null.normalize_mut();
        no_zeroes(&mut near_null);
        normalize(&mut near_null, mat);

        let ortho_check: Vec<f64> = near_null_history
            .iter()
            .map(|old| old.dot(&near_null))
            .collect();
        near_null_history.push(near_null.clone());
        trace!("Near null component inner product with history: {ortho_check:?}");

        while modularity_matching_add_level(&near_null, coarsening_factor, &mut hierarchy) {
            near_null = {
                let current_a = hierarchy.get_matrices().last().unwrap_or(&hierarchy[0]);
                let dim = current_a.nrows();
                l1 = L1::new(&current_a);
                let zeros = DVector::from(vec![0.0; dim]);
                let mut near_null = random_vec(dim);
                // TODO: maybe go untill stall instead
                let _ = pcg(
                    &current_a,
                    &zeros,
                    &mut near_null,
                    40,
                    1e-8,
                    &mut l1,
                    None,
                );
                no_zeroes(&mut near_null);
                normalize(&mut near_null, &current_a);
                near_null
            };
            //near_null.normalize_mut();
            //no_zeroes(&mut near_null);
        }

        let ml1 = Multilevel::<PcgL1>::new(hierarchy);
        preconditioner.push(Box::new(ml1));

        near_null = random_vec(dim);
        if let Some((convergence_rate, convergence_history)) =
            find_near_null(mat, &mut preconditioner, &mut near_null)
        {
            let trace = Scatter::new(
                (1..=convergence_history.len()).collect(),
                convergence_history,
            );
            plot.add_trace(trace);
            let layout = Layout::new()
                .title("Preconditioner Testing".into())
                .y_axis(
                    Axis::new()
                        .title("Relative Error in A-norm".into())
                        .type_(AxisType::Log),
                )
                .x_axis(Axis::new().title("Iteration".into()));
            plot.set_layout(layout);
            // TODO: add some metadata to output with unique name
            plot.write_html("data/output/out.html");
            if convergence_rate < 0.15 || preconditioner.components().len() == 25 {
                return preconditioner;
            }
        } else {
            return preconditioner;
        }
    }
}

// TODO on tester make *relative* error be convergence test
//      additive version x += 1/components (Bk^-1 * r)
//      one residual per iteration, doesnt change between components
//      try throwing early components **** dangerous
fn stationary_composite(
    mat: &CsrMatrix<f64>,
    iterate: &mut DVector<f64>,
    iterations: usize,
    composite_preconditioner: &mut Composite,
) {
    for _ in 0..iterations {
        let mut residual = -1.0 * (mat * &*iterate);
        composite_preconditioner.apply(&mut residual);
        *iterate += &residual;
    }
}

fn find_near_null<'a>(
    mat: &'a CsrMatrix<f64>,
    composite_preconditioner: &mut Composite,
    near_null: &mut DVector<f64>,
) -> Option<(f64, Vec<f64>)> {
    trace!("searching for near null");
    let mut iter = 0;
    let start = Instant::now();
    let mut last_log = start;
    let log_interval = Duration::from_secs(2);
    let starting_error = near_null.dot(&(mat * &*near_null));
    let mut convergence_history: Vec<f64> = Vec::new();
    let mut old_error_norm = f64::MAX;
    let mut convergence_rate_per_second;
    let mut convergence_rate_per_iter;
    let zeros = DVector::from(vec![0.0; near_null.nrows()]);

    loop {
        //stationary_composite(mat, near_null, 1, composite_preconditioner);
        pcg(
            &mat,
            &zeros,     //rhs
            near_null, //initial
            1,
            1e-16,
            composite_preconditioner,
            None,
        );
        iter += 1;
        let current_error = near_null.dot(&(mat * &*near_null));
        if current_error < 1e-16 {
            warn!("find_near_null converged during search");
            return None;
        }
        let error_ratio = current_error / old_error_norm;
        old_error_norm = current_error;

        let now = Instant::now();
        let elapsed = (now - start).as_millis();
        let elapsed_secs = elapsed as f64 / 1000.0;
        let convergence = current_error / starting_error;
        let relative_error = convergence.sqrt();
        convergence_history.push(relative_error);
        convergence_rate_per_second = convergence.powf(1.0 / (elapsed_secs * 2.0));
        convergence_rate_per_iter = convergence.powf(1.0 / (iter as f64 * 2.0));

        if now - last_log > log_interval {
            trace!(
                "iteration {}:\n\ttotal search time: {}s\n\tconvergence per second: {:.3}\n\tconvergence per iteration: {:.3}\n\terror ratio to previous iteration: {:.3}\n\trelative error (A-norm): {:.3e}",
                iter,
                elapsed_secs,
                convergence_rate_per_second,
                convergence_rate_per_iter,
                error_ratio,
                relative_error 
            );
            last_log = now;
        }

        if error_ratio > 0.94 || iter > 20 {
            info!(
                "{} components:\n\tconvergence rate stabilized at {:.3}/iter on iteration {} after {} seconds\n\tsquared error: {:.3e}\n\trelative error: {:.3e}",
                composite_preconditioner.components().len(), convergence_rate_per_iter, iter, (now - start).as_secs(), current_error, convergence.powf(0.5)
            );
            return Some((convergence_rate_per_iter, convergence_history));
        }
    }
}

fn no_zeroes(near_null: &mut DVector<f64>) {
    let mut rng = thread_rng();
    let epsilon = 1.0 / (near_null.nrows() as f64);
    let threshold = 1.0e-10_f64;
    let range = rand::distributions::Uniform::new(threshold, epsilon);

    let mut counter = 0;
    for x in near_null.iter_mut().filter(|x| x.abs() < threshold) {
        let val = range.sample(&mut rng);
        if rng.gen() {
            *x = val;
        } else {
            *x = -val;
        }
        counter += 1
    }
    if counter > 0 {
        warn!("perturbed {} of {} elements", counter, near_null.nrows());
    }
}

/*
fn composite_tester(
    mat: &CsrMatrix<f64>,
    steps: usize,
    composite_preconditioner: &mut Composite,
) -> f64 {
    trace!("testing convergence...");
    let test_runs = 2;
    let mut avg_per_second = 0.0;
    let mut avg_per_iter = 0.0;
    let dim = mat.nrows();
    // 1 / (2.0 * however many steps done after test)
    let test_seconds = 5;
    let distribution = Uniform::new(-1e-3_f64, 1e-3_f64);
    let mut rng = thread_rng();

    for _ in 0..test_runs {
        let mut iterate: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
        //stationary_composite(mat, &mut iterate, steps, composite_preconditioner);
        let _ = crate::solver::pcg(
            mat,
            &DVector::zeros(dim),
            &mut iterate,
            10,
            1e-16,
            &mut Sgs::new(mat),
            None,
        );
        let start = Instant::now();
        let time = Duration::from_secs(5);
        let starting_error_norm = iterate.dot(&(mat * &iterate));
        let mut completed_iterations = 0;
        while start.elapsed() < time {
            stationary_composite(mat, &mut iterate, 1, composite_preconditioner);
            completed_iterations += 1;
        }
        if starting_error_norm < 1e-16 {
            warn!("tester converged in less than number of tests ({steps})");
            return 0.0;
        }
        let final_error_norm = iterate.dot(&(mat * &iterate));
        let convergence = final_error_norm / starting_error_norm;
        let convergence_rate_per_second = convergence.powf(1.0 / (test_seconds as f64 * 2.0));
        let convergence_rate_per_iter = convergence.powf(1.0 / (completed_iterations as f64 * 2.0));
        avg_per_second += convergence_rate_per_second;
        avg_per_iter += convergence_rate_per_iter;
    }
    avg /= test_runs as f64;

    avg
}
*/

#[cfg(test)]
mod tests {
    use crate::{adaptive::build_adaptive, preconditioner::Preconditioner, random_vec};
    use nalgebra_sparse::CsrMatrix;
    use test_generator::test_resources;

    fn test_symmetry(preconditioner: &mut dyn Preconditioner, dim: usize) {
        for _ in 0..5 {
            let u = random_vec(dim);
            let v = random_vec(dim);
            let mut preconditioned_v = v.clone();
            let mut preconditioned_u = u.clone();
            preconditioner.apply(&mut preconditioned_v);
            preconditioner.apply(&mut preconditioned_u);

            let left: f64 = u.dot(&preconditioned_v);
            let right: f64 = v.dot(&preconditioned_u);
            let difference = (left - right).abs() / (left + right).abs();
            assert!(
                difference < 1e-3,
                "\nLeft and right didn't match\nleft: {}\nright: {}\nrelative difference: {:+e}\n",
                left,
                right,
                difference
            );
        }
    }

    #[test_resources("test_matrices/unit_tests/*")]
    fn test_symmetry_adaptive(mat_path: &str) {
        let mut mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(mat_path).unwrap(),
        );

        let norm = mat
            .values()
            .iter()
            .fold(0.0_f64, |acc, x| acc + x * x)
            .sqrt();
        mat /= norm;
        let mat = mat;
        let dim = mat.nrows();
        let mut preconditioner = build_adaptive(&mat);
        test_symmetry(&mut preconditioner, dim);
    }
}
