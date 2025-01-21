//! This module contains the code to construct the adaptive preconditioner.

use crate::{
    partitioner::{modularity_matching_add_level, Hierarchy, InterpolationType},
    preconditioner::{Composite, LinearOperator, Multilevel, SmootherType, L1},
    solver::{Iterative, IterativeMethod},
    utils::{inner_product, norm, random_vec},
};
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct AdaptiveBuilder {
    mat: Arc<CsrMatrix<f64>>,
    coarsening_factor: f64,
    max_level: Option<usize>,
    target_convergence: Option<f64>,
    max_components: Option<usize>,
    test_iters: Option<usize>,
    // TODO max test duration?
    solve_coarsest_exactly: bool,
    smoothing_steps: usize,
    smoother_type: SmootherType,
    interpolation_type: InterpolationType,
}

impl AdaptiveBuilder {
    pub fn new(mat: Arc<CsrMatrix<f64>>) -> Self {
        AdaptiveBuilder {
            mat,
            coarsening_factor: 8.0,
            max_level: None,
            target_convergence: None,
            max_components: Some(10),
            test_iters: None,
            solve_coarsest_exactly: true,
            smoothing_steps: 1,
            smoother_type: SmootherType::BlockGaussSeidel,
            interpolation_type: InterpolationType::SmoothedAggregation(1),
        }
    }

    pub fn with_matrix(mut self, mat: Arc<CsrMatrix<f64>>) -> Self {
        self.mat = mat;
        self
    }

    pub fn with_coarsening_factor(mut self, coarsening_factor: f64) -> Self {
        self.coarsening_factor = coarsening_factor;
        self
    }

    pub fn with_max_level(mut self, max_level: usize) -> Self {
        self.max_level = Some(max_level);
        self
    }

    pub fn without_max_level(mut self) -> Self {
        self.max_level = None;
        self
    }

    pub fn with_target_convergence(mut self, target_convergence: f64) -> Self {
        self.target_convergence = Some(target_convergence);
        self
    }

    pub fn without_target_convergence(mut self) -> Self {
        self.target_convergence = None;
        self
    }

    pub fn with_max_components(mut self, max_components: usize) -> Self {
        self.max_components = Some(max_components);
        self
    }

    pub fn without_max_components(mut self) -> Self {
        self.max_components = None;
        self
    }

    pub fn with_max_test_iters(mut self, test_iters: usize) -> Self {
        self.test_iters = Some(test_iters);
        self
    }

    pub fn without_max_test_iters(mut self) -> Self {
        self.test_iters = None;
        self
    }

    pub fn with_smoother(mut self, smoother: SmootherType) -> Self {
        self.smoother_type = smoother;
        self
    }

    pub fn with_smoothing_steps(mut self, steps: usize) -> Self {
        self.smoothing_steps = steps;
        self
    }

    pub fn solve_coarsest_exactly(mut self) -> Self {
        self.solve_coarsest_exactly = true;
        self
    }

    pub fn smooth_coarsest(mut self) -> Self {
        self.solve_coarsest_exactly = false;
        self
    }

    pub fn with_interpolator(mut self, interpolation: InterpolationType) -> Self {
        self.interpolation_type = interpolation;
        self
    }

    //TODO from yaml config??

    //TODO log intervals and max time?
    pub fn build(&self) -> (Composite, Vec<Vec<f64>>, Vec<DVector<f64>>) {
        let mut preconditioner = Composite::new(self.mat.clone());

        let dim = self.mat.nrows();
        let mut near_null_history = Vec::<DVector<f64>>::new();
        let mut test_data = Vec::new();

        // Find initial near null to get the iterations started
        let fine_l1 = Arc::new(L1::new(&self.mat));
        info!("built smoother");
        let guess: DVector<f64> = DVector::from_element(dim, 1.0);
        let stationary = Iterative::new(self.mat.clone(), Some(guess))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_max_iter(3)
            .with_preconditioner(fine_l1.clone());
        let zeros = DVector::from(vec![0.0; dim]);
        let mut near_null = stationary.apply(&zeros);

        loop {
            trace!("Near-Null score: {:.2e}", norm(&near_null, &*self.mat));

            // Sanity check that each near null is orthogonal to the last.
            // Could move into test suite down the line.
            let ortho_check: String = near_null_history
                .iter()
                .map(|old| format!("{:.1e}, ", inner_product(old, &near_null, &self.mat)))
                .collect();
            near_null_history.push(near_null.clone());
            if !near_null_history.is_empty() {
                trace!("Near null component inner product with history: {ortho_check}");
            }

            let mut hierarchy = Hierarchy::new(self.mat.clone());
            let mut levels = 1;

            loop {
                near_null = modularity_matching_add_level(
                    &near_null,
                    self.coarsening_factor,
                    &mut hierarchy,
                    self.interpolation_type,
                );

                levels += 1;
                if let Some(max_level) = self.max_level {
                    if levels == max_level {
                        break;
                    }
                }
                if hierarchy
                    .get_matrices()
                    .last()
                    .unwrap_or(&hierarchy.get_mat(0))
                    .nrows()
                    < 100
                {
                    break;
                }

                let current_a = hierarchy
                    .get_matrices()
                    .last()
                    .unwrap_or(&hierarchy.get_mat(0))
                    .clone();

                let dim = current_a.nrows();
                let l1 = Arc::new(L1::new(&current_a));
                let zeros = DVector::from(vec![0.0; dim]);

                let stationary = Iterative::new(current_a.clone(), Some(near_null))
                    .with_solver(IterativeMethod::StationaryIteration)
                    .with_max_iter(3)
                    .with_preconditioner(l1.clone());
                near_null = stationary.apply(&zeros);
                //normalize(&mut near_null, &current_a);
                near_null.normalize_mut();
            }
            //hierarchy.consolidate(self.coarsening_factor);
            info!("Hierarchy info*: {:?}", hierarchy);

            let ml1 = Arc::new(Multilevel::new(
                hierarchy,
                self.solve_coarsest_exactly,
                self.smoother_type,
                self.smoothing_steps,
            ));
            trace!("Multilevel pc constructed");
            preconditioner.push(ml1);

            near_null = random_vec(dim);
            let (convergence_rate, convergence_history) = find_near_null(
                self.mat.clone(),
                &preconditioner,
                &mut near_null,
                self.test_iters,
            );

            for (i, (cf, cf_next)) in convergence_history
                .iter()
                .copied()
                .zip(convergence_history.iter().skip(1).copied())
                .enumerate()
            {
                if cf > cf_next || cf >= 1.0 {
                    error!("Monotonicity or convergence properties violated in homogeneous problem at iter: {}, cf_i: {:.2}, cf_i+1: {:.2}\nConvergence history: {:?}", i, cf, cf_next, convergence_history);
                }
            }

            test_data.push(convergence_history);
            //plot_convergence_history("in_progress", &test_data, 1);
            if let Some(max_components) = self.max_components {
                if preconditioner.components().len() >= max_components {
                    return (preconditioner, test_data, near_null_history);
                }
            }
            if let Some(target_convergence) = self.target_convergence {
                if convergence_rate < target_convergence {
                    return (preconditioner, test_data, near_null_history);
                }
            }
        }
    }
}

fn find_near_null(
    mat: Arc<CsrMatrix<f64>>,
    composite_preconditioner: &Composite,
    near_null: &mut DVector<f64>,
    test_iters: Option<usize>,
) -> (f64, Vec<f64>) {
    let mut iter = 0;
    let max_iter = test_iters.unwrap_or(usize::MAX);
    let start = Instant::now();
    let mut last_log = start;
    let log_interval = Duration::from_secs(10);
    let zeros = DVector::from(vec![0.0; near_null.nrows()]);
    let mut old_convergence_factor = 0.0;
    let mut history = Vec::new();

    loop {
        near_null.normalize_mut();
        let stationary = Iterative::new(mat.clone(), Some(near_null.clone()))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Arc::new(composite_preconditioner.clone()))
            .with_max_iter(1);
        *near_null = stationary.apply(&zeros);
        iter += 1;
        let convergence_factor = near_null.norm();
        history.push(convergence_factor);

        let now = Instant::now();
        let elapsed = (now - start).as_millis();
        let elapsed_secs = elapsed as f64 / 1000.0;

        let cycles = ((composite_preconditioner.components().len() * 2) - 1) as f64;
        if now - last_log > log_interval {
            trace!(
                "iteration {}:\n\ttotal search time: {:.0}s\n\tConvergence Factor: {:.3}\n\t CF per cycle: {:.3}",
                iter,
                elapsed_secs,
                convergence_factor,
                convergence_factor.powf(1.0 / cycles)
            );
            last_log = now;
        }

        if old_convergence_factor / convergence_factor > 0.999 || iter >= max_iter {
            info!(
                "{} components:\n\tconvergence factor: {:.3}\n\tconvergence factor per cycle: {:.3}\n\tsearch iters: {}",
                composite_preconditioner.components().len(),
                convergence_factor,
                convergence_factor.powf(1.0 / cycles),
                iter
            );
            return (convergence_factor, history);
        }
        old_convergence_factor = convergence_factor;
    }
}

/*
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
    */

//#[test_resources("test_matrices/unit_tests/*")]
/*
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

*/
