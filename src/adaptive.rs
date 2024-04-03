//! This module contains the code to construct the adaptive preconditioner.

use crate::{
    io::plot_convergence_history,
    partitioner::{modularity_matching, modularity_matching_add_level, Hierarchy},
    preconditioner::{Composite, LinearOperator, Multilevel, L1},
    solver::{Iterative, IterativeMethod},
    utils::{norm, random_vec},
};
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

pub struct AdaptiveBuilder {
    mat: std::rc::Rc<CsrMatrix<f64>>,
    coarsening_factor: f64,
    max_level: Option<usize>,
    target_convergence: Option<f64>,
    max_components: Option<usize>,
    test_iters: Option<usize>,
    // TODO max test duration?
    project_first_only: bool,
}

impl AdaptiveBuilder {
    pub fn new(mat: Rc<CsrMatrix<f64>>) -> Self {
        AdaptiveBuilder {
            mat,
            coarsening_factor: 8.0,
            max_level: None,
            target_convergence: None,
            max_components: Some(10),
            test_iters: None,
            project_first_only: false,
        }
    }

    pub fn with_matrix(mut self, mat: Rc<CsrMatrix<f64>>) -> Self {
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

    pub fn with_project_first_only(mut self) -> Self {
        self.project_first_only = true;
        self
    }

    pub fn with_project_all_levels(mut self) -> Self {
        self.project_first_only = false;
        self
    }

    //TODO log intervals and max time?
    pub fn build(&self) -> (Composite, Vec<Vec<f64>>, Vec<DVector<f64>>) {
        let mut preconditioner = Composite::new(self.mat.clone());

        let dim = self.mat.nrows();
        let mut near_null_history = Vec::<DVector<f64>>::new();
        let mut test_data = Vec::new();
        //let solve_exactly = self.max_level.unwrap_or(0) == 2;

        // TODO arg all 3 of these?
        let solve_exactly = true;
        //let smoother = crate::preconditioner::SmootherType::GaussSeidel;
        let smoother = crate::preconditioner::SmootherType::L1;
        let sweeps = 5;

        // Find initial near null to get the iterations started
        let fine_l1 = Rc::new(L1::new(&self.mat));
        //let mut near_null: DVector<f64> = DVector::from_element(dim, 1.0); //.normalize();
        let guess: DVector<f64> = DVector::from_element(dim, 1.0); //.normalize();
        let stationary = Iterative::new(self.mat.clone(), Some(guess))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_tolerance(1e-3)
            .with_max_iter(1000)
            //.with_log_interval(LogInterval::Iterations(1001))
            .with_preconditioner(fine_l1.clone());
        let zeros = DVector::from(vec![0.0; dim]);
        let mut near_null = stationary.apply(&zeros);

        loop {
            //normalize(&mut near_null, &self.mat);
            near_null.normalize_mut();
            let test = &*self.mat * &near_null;
            trace!("Near-Null score: {:.2e}", test.norm());

            // Sanity check that each near null is orthogonal to the last.
            // Could move into test suite down the line.
            let ortho_check: String = near_null_history
                .iter()
                //.map(|old| format!("{:.1e}, ", inner_product(old, &near_null, &self.mat)))
                .map(|old| format!("{:.1e}, ", near_null.dot(old)))
                .collect();
            near_null_history.push(near_null.clone());
            if !near_null_history.is_empty() {
                trace!("Near null component inner product with history: {ortho_check}");
            }

            let hierarchy = if self.project_first_only {
                modularity_matching(self.mat.clone(), &near_null, self.coarsening_factor)
            } else {
                let mut hierarchy = Hierarchy::new(self.mat.clone());
                let mut levels = 1;
                while modularity_matching_add_level(
                    &near_null,
                    self.coarsening_factor,
                    &mut hierarchy,
                ) {
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
                    near_null = {
                        let current_a = hierarchy
                            .get_matrices()
                            .last()
                            .unwrap_or(&hierarchy.get_mat(0))
                            .clone();

                        let dim = current_a.nrows();
                        let start = hierarchy.get_interpolations().last().unwrap() * near_null;
                        //let start = DVector::from(vec![1.0; dim]);
                        let l1 = Rc::new(L1::new(&current_a));
                        let zeros = DVector::from(vec![0.0; dim]);

                        let stationary = Iterative::new(current_a.clone(), Some(start))
                            .with_solver(IterativeMethod::StationaryIteration)
                            .with_tolerance(1e-4)
                            .with_max_iter(1000)
                            .with_preconditioner(l1);
                        let mut near_null = stationary.apply(&zeros);
                        //normalize(&mut near_null, &current_a);
                        near_null.normalize_mut();
                        near_null
                    };
                }
                hierarchy
            };
            info!("Hierarchy info: {:?}", hierarchy);

            let ml1 = Rc::new(Multilevel::new(
                hierarchy,
                Some(fine_l1.clone()),
                //None,
                solve_exactly,
                smoother,
                sweeps,
            ));
            preconditioner.push(ml1);
            if let Some(max_components) = self.max_components {
                if preconditioner.components().len() >= max_components {
                    return (preconditioner, test_data, near_null_history);
                }
            }

            near_null = random_vec(dim);
            if let Some((convergence_rate, convergence_history)) = find_near_null(
                self.mat.clone(),
                &preconditioner,
                &mut near_null,
                self.test_iters,
            ) {
                test_data.push(convergence_history);
                plot_convergence_history("in_progress", &test_data, 1);
                if let Some(target_convergence) = self.target_convergence {
                    if convergence_rate < target_convergence {
                        return (preconditioner, test_data, near_null_history);
                    }
                }
            } else {
                // TODO keep last convergence history if tester converges (low prio)
                return (preconditioner, test_data, near_null_history);
            }
        }
    }
}

//TODO log intervals and max time?
fn find_near_null(
    mat: Rc<CsrMatrix<f64>>,
    composite_preconditioner: &Composite,
    near_null: &mut DVector<f64>,
    test_iters: Option<usize>,
) -> Option<(f64, Vec<f64>)> {
    trace!("searching for near null");
    let mut iter = 0;
    let start = Instant::now();
    let mut last_log = start;
    let log_interval = Duration::from_secs(10);
    let starting_error = norm(&near_null, &*mat);
    let mut convergence_history: Vec<f64> = Vec::new();
    let mut old_error_norm = f64::MAX;
    let mut convergence_rate_per_second;
    let mut convergence_rate_per_iter;
    let zeros = DVector::from(vec![0.0; near_null.nrows()]);
    let mut error_ratio = f64::MAX;

    loop {
        let stationary = Iterative::new(mat.clone(), Some(near_null.clone()))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Rc::new(composite_preconditioner.clone()))
            .with_max_iter(1)
            .with_tolerance(1e-16);
        *near_null = stationary.apply(&zeros);
        iter += 1;
        let current_error = norm(&near_null, &*mat);
        let previous_error_ratio = error_ratio;
        error_ratio = current_error / old_error_norm;
        old_error_norm = current_error;

        // TODO fix this maybe
        let now = Instant::now();
        let elapsed = (now - start).as_millis();
        let elapsed_secs = elapsed as f64 / 1000.0;

        let convergence = current_error / starting_error;
        if convergence < 1e-12 {
            warn!("find_near_null converged during search");
            return None;
        }
        convergence_history.push(convergence);
        convergence_rate_per_second = convergence.powf(1.0 / elapsed_secs);
        convergence_rate_per_iter = convergence.powf(1.0 / iter as f64);
        // TODO use rayleigh quotient?
        let ratio_change = (previous_error_ratio - error_ratio).abs();

        if now - last_log > log_interval {
            trace!(
                "iteration {}:\n\ttotal search time: {:.0}s\n\tconvergence per second: {:.3}\n\tconvergence per iteration: {:.3}\n\terror ratio to previous iteration: {:.3}\n\trelative error (A-norm): {:.2e}\n\tratio change: {:.2e}",
                iter,
                elapsed_secs,
                convergence_rate_per_second,
                convergence_rate_per_iter,
                error_ratio,
                convergence,
                ratio_change
            );
            last_log = now;
        }

        if let Some(iters) = test_iters {
            if iter == iters {
                info!(
                    "{} components:\n\tconvergence rate: {:.3}/iter on iteration {} after {} seconds\n\tsquared error: {:.3e}\n\trelative error: {:.2e}",
                    composite_preconditioner.components().len(), convergence_rate_per_iter, iter, (now - start).as_secs(), current_error, convergence
                );
                return Some((convergence_rate_per_iter, convergence_history));
            }
            // Basically when the convergence rate stops slowing down we have hit the asymptotic
            // rate (TODO: maybe make sure the last few error ratios are all about the same? since
            // we are doing stationary iterations though this might not really matter)
        } else if ratio_change < 1e-3 {
            //} else if error_ratio > 0.90 {
            info!(
                "{} components:\n\tconvergence rate: {:.3}/iter on iteration {} after {} seconds\n\tsquared error: {:.3e}\n\trelative error: {:.2e}\n\tasymptotic rate: {:.3}",
                composite_preconditioner.components().len(), convergence_rate_per_iter, iter, (now - start).as_secs(), current_error, convergence, error_ratio
            );
            return Some((convergence_rate_per_iter, convergence_history));
        }
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
