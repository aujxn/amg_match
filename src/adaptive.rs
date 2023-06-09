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

pub fn build_adaptive<'a>(mat: &'a CsrMatrix<f64>, coarsening_factor: f64) -> Composite<'a> {
    let mut plot = Plot::new();
    let dim = mat.nrows();

    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut near_null = random_vec(dim);
    let mut preconditioner = Composite::new(mat, Vec::new(), CompositeType::Sequential);
    let mut l1 = L1::new(mat);
    let _ = pcg(mat, &zeros, &mut near_null, 30, 1e-16, &mut l1, None);
    no_zeroes(&mut near_null);
    normalize(&mut near_null, mat);

    let hierarchy = modularity_matching(&mat, &near_null, coarsening_factor);
    let ml1 = Multilevel::<PcgL1>::new(hierarchy);
    preconditioner.push(Box::new(ml1));

    loop {
        near_null = random_vec(dim);
        let _ = pcg(mat, &zeros, &mut near_null, 10, 1e-16, &mut l1, None);

        if let Some((convergence_rate, convergence_history)) =
            find_near_null(mat, &mut preconditioner, &mut near_null)
        {
            add_trace(&mut plot, convergence_history);
            no_zeroes(&mut near_null);
            normalize(&mut near_null, mat);

            let hierarchy = modularity_matching(&mat, &near_null, coarsening_factor);
            let ml1 = Multilevel::<PcgL1>::new(hierarchy);
            preconditioner.push(Box::new(ml1));
            if convergence_rate < 0.15 || preconditioner.components().len() == 8 {
                return preconditioner;
            }
        } else {
            return preconditioner;
        }
    }
}

/// Alternate implementation of construction of the PC where a near null component is
/// found for every level in the hierarchy and projected into the interpolation matrix.
pub fn build_adaptive_new<'a>(mat: &'a CsrMatrix<f64>, coarsening_factor: f64) -> Composite<'a> {
    let mut preconditioner = Composite::new(mat, Vec::new(), CompositeType::Sequential);

    let mut plot = Plot::new();
    let dim = mat.nrows();
    let mut near_null_history = Vec::<DVector<f64>>::new();

    // Find initial near null to get the iterations started
    let mut l1 = L1::new(mat);
    let zeros = DVector::from(vec![0.0; dim]);
    let mut near_null = random_vec(dim);
    let _ = pcg(mat, &zeros, &mut near_null, 50, 1e-16, &mut l1, None);

    loop {
        let mut hierarchy = Hierarchy::new(mat);
        no_zeroes(&mut near_null);
        normalize(&mut near_null, mat);

        // Sanity check that each near null is orthogonal to the last.
        // Could move into test suite down the line.
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
        }

        let ml1 = Multilevel::<PcgL1>::new(hierarchy);
        preconditioner.push(Box::new(ml1));

        near_null = random_vec(dim);
        if let Some((convergence_rate, convergence_history)) =
            find_near_null(mat, &mut preconditioner, &mut near_null)
        {
            add_trace(&mut plot, convergence_history);
            if convergence_rate < 0.15 || preconditioner.components().len() == 25 {
                return preconditioner;
            }
        } else {
            return preconditioner;
        }
    }
}

fn add_trace(plot: &mut Plot, data: Vec<f64>) {
    let trace = Scatter::new(
        (1..=data.len()).collect(),
        data,
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
    plot.write_html("data/out/out.html");
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
    // TODO these values could probably be tuned. idk what they should be
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
