//! This module contains the code to construct the adaptive preconditioner.

use ndarray_linalg::Norm;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::{
    hierarchy::Hierarchy,
    interpolation::InterpolationType,
    partitioner::{metis_n, modularity_matching_partition},
    preconditioner::{build_smoother, Composite, LinearOperator, Multilevel, SmootherType, L1},
    solver::{Iterative, IterativeMethod},
    utils::{format_duration, inner_product, norm, normalize},
    CsrMatrix, Vector,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct AdaptiveBuilder {
    mat: Arc<CsrMatrix>,
    coarsening_factor: f64,
    max_level: Option<usize>,
    mu: usize,
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
    pub fn new(mat: Arc<CsrMatrix>) -> Self {
        AdaptiveBuilder {
            mat,
            coarsening_factor: 8.0,
            max_level: None,
            target_convergence: None,
            max_components: Some(10),
            mu: 1,
            test_iters: None,
            solve_coarsest_exactly: true,
            smoothing_steps: 1,
            smoother_type: SmootherType::BlockGaussSeidel,
            interpolation_type: InterpolationType::SmoothedAggregation((1, 0.66)),
        }
    }

    pub fn with_matrix(mut self, mat: Arc<CsrMatrix>) -> Self {
        self.mat = mat;
        self
    }

    pub fn with_coarsening_factor(mut self, coarsening_factor: f64) -> Self {
        self.coarsening_factor = coarsening_factor;
        self
    }

    pub fn cycle_type(mut self, mu: usize) -> Self {
        self.mu = mu;
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
    pub fn build(&self) -> (Composite, Vec<Vec<f64>>, Vec<Vector>) {
        let mut preconditioner = Composite::new(self.mat.clone());

        let dim = self.mat.rows();
        let mut near_null_history = Vec::<Vector>::new();
        let mut test_data = Vec::new();

        // Find initial near null to get the iterations started
        let fine_l1 = Arc::new(L1::new(&self.mat));
        let guess: Vector = Vector::from_elem(dim, 1.0);
        let stationary = Iterative::new(self.mat.clone(), Some(guess))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_max_iter(3)
            .with_preconditioner(fine_l1.clone());
        let zeros = Vector::from(vec![0.0; dim]);
        let mut near_null: Vector = stationary.apply(&zeros);

        loop {
            let almost0 = &*self.mat * &near_null;
            let score = almost0.norm();
            trace!("Near-Null score: {:.2e}", score);

            // Sanity check that each near null is orthogonal to the last.
            // Could move into test suite down the line.
            normalize(&mut near_null, &self.mat);
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
                near_null = hierarchy.add_level(
                    &near_null,
                    self.coarsening_factor,
                    self.interpolation_type,
                );

                levels += 1;
                if let Some(max_level) = self.max_level {
                    if levels == max_level {
                        break;
                    }
                }
                if hierarchy
                    .get_coarse_mats()
                    .last()
                    .unwrap_or(&hierarchy.get_mat(0))
                    .rows()
                    < 100
                {
                    break;
                }

                let current_a = hierarchy
                    .get_coarse_mats()
                    .last()
                    .unwrap_or(&hierarchy.get_mat(0))
                    .clone();

                //let r = metis_n(&near_null, current_a.clone(), 16);
                //let r = modularity_matching_partition(current_a.clone(), &near_null, 64.0, Some(64));
                //let coarse_smoother = build_smoother(current_a.clone(), self.smoother_type, r.into(), false);
                let coarse_smoother = Arc::new(L1::new(&current_a));
                find_near_null_coarse(current_a, coarse_smoother, &mut near_null, 5);
                /*
                let dim = current_a.rows();
                let zeros = Vector::from(vec![0.0; dim]);

                let stationary = Iterative::new(current_a.clone(), Some(near_null))
                    .with_solver(IterativeMethod::StationaryIteration)
                    .with_max_iter(3)
                    .with_preconditioner(l1.clone());
                near_null = stationary.apply(&zeros);
                */
            }
            //hierarchy.consolidate(self.coarsening_factor);
            info!("Hierarchy info: {:?}", hierarchy);

            let ml1 = Arc::new(Multilevel::new(
                hierarchy,
                self.solve_coarsest_exactly,
                self.smoother_type,
                self.smoothing_steps,
                self.mu,
            ));
            preconditioner.push(ml1);

            near_null = Vector::random(dim, Uniform::new(-1., 1.));
            let (convergence_rate, convergence_history) = find_near_null(
                self.mat.clone(),
                &preconditioner,
                &mut near_null,
                self.test_iters,
            );

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
    mat: Arc<CsrMatrix>,
    composite_preconditioner: &Composite,
    near_null: &mut Vector,
    test_iters: Option<usize>,
) -> (f64, Vec<f64>) {
    let mut iter = 0;
    let max_iter = test_iters.unwrap_or(usize::MAX);
    let start = Instant::now();
    let mut last_log = start;
    let log_interval = Duration::from_secs(10);
    let zeros = Vector::from(vec![0.0; near_null.len()]);
    let mut old_convergence_factor = 0.0;
    let mut history = Vec::new();

    loop {
        normalize(near_null, &mat);
        //near_null.normalize_mut();
        let stationary = Iterative::new(mat.clone(), Some(near_null.clone()))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Arc::new(composite_preconditioner.clone()))
            .with_max_iter(1);
        *near_null = stationary.apply(&zeros);
        iter += 1;
        //let convergence_factor = near_null.norm();
        let convergence_factor = norm(near_null, &mat);
        if convergence_factor > 1.0 {
            error!(
                "Not convergent method! Tester convergence factor: {:.2} on iteration {}",
                convergence_factor, iter
            );
            //maybe panic?
        }

        if convergence_factor < old_convergence_factor {
            warn!("Monotonicity properties violated in tester at iter: {}, cf_i: {:.2}, cf_i-1: {:.2}", iter, convergence_factor, old_convergence_factor);
        }
        history.push(convergence_factor);

        let now = Instant::now();
        let elapsed = now - start;
        let elapsed_secs = elapsed.as_millis() as f64 / 1000.0;

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

        //if old_convergence_factor / convergence_factor > 0.999 || iter >= max_iter {
        if iter >= max_iter {
            info!(
                "{} components:\n\tconvergence factor: {:.3}\n\tconvergence factor per cycle: {:.3}\n\tsearch iters: {}\n\tsearch time: {}",
                composite_preconditioner.components().len(),
                convergence_factor,
                convergence_factor.powf(1.0 / cycles),
                iter,
                format_duration(&elapsed)
            );
            return (convergence_factor, history);
        }
        old_convergence_factor = convergence_factor;
    }
}

fn find_near_null_coarse(
    mat: Arc<CsrMatrix>,
    pc: Arc<dyn LinearOperator + Send + Sync>,
    near_null: &mut Vector,
    max_iter: usize,
) -> f64 {
    let mut iter = 0;
    let zeros = Vector::from(vec![0.0; near_null.len()]);
    let mut old_convergence_factor = 0.0;

    loop {
        normalize(near_null, &mat);
        let stationary = Iterative::new(mat.clone(), Some(near_null.clone()))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(pc.clone())
            .with_max_iter(1);
        *near_null = stationary.apply(&zeros);
        iter += 1;
        let convergence_factor = norm(near_null, &mat);
        if convergence_factor > 1.0 {
            error!(
                "Not convergent method! Tester convergence factor: {:.2} on iteration {}",
                convergence_factor, iter
            );
        }
        if convergence_factor < old_convergence_factor {
            warn!("Monotonicity properties violated in tester at iter: {}, cf_i: {:.2}, cf_i-1: {:.2}", iter, convergence_factor, old_convergence_factor);
        }

        if iter >= max_iter {
            return convergence_factor;
        }
        old_convergence_factor = convergence_factor;
    }
}
