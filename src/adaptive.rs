//! This module contains the code to construct the adaptive preconditioner.

use crate::{
    hierarchy::{Hierarchy, HierarchyData},
    interpolation::{smoothed_aggregation2, InterpolationType},
    output_path,
    parallel_ops::spmv,
    partitioner::{BlockReductionStrategy, PartitionBuilder},
    preconditioner::{
        BlockSmootherType, Composite, LinearOperator, Multilevel, SmootherType,
        SymmetricGaussSeidel, L1,
    },
    solver::{Iterative, IterativeMethod},
    utils::{format_duration, inner_product, norm, normalize, orthonormalize_mgs},
    CsrMatrix, Matrix, Vector,
};
use ndarray_linalg::Norm;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{fmt::Write as _, fs::File, io::Write, usize};

#[derive(Serialize, Deserialize)]
pub struct AdaptiveBuilder {
    #[serde(skip_serializing)]
    mat: Arc<CsrMatrix>,
    coarsening_factor: f64,
    max_level: Option<usize>,
    mu: usize,
    target_convergence: Option<f64>,
    max_components: Option<usize>,
    test_iters: Option<usize>,
    // TODO max test duration?
    solve_coarsest_exactly: bool,
    // coarsest_size: usize,
    smoothing_steps: usize,
    smoother_type: SmootherType,
    interpolation_type: InterpolationType,
    block_size: usize,
    near_null_dim: usize,
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
            smoother_type: SmootherType::DiagonalCompensatedBlock(
                BlockSmootherType::GaussSeidel,
                16,
            ),
            interpolation_type: InterpolationType::SmoothedAggregation((1, 0.66)),
            block_size: 1,
            near_null_dim: 1,
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

    pub fn set_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn set_near_null_dim(mut self, dim: usize) -> Self {
        self.near_null_dim = dim;
        self
    }

    //TODO from yaml config??

    //TODO log intervals and max time?
    pub fn build(&self) -> (Composite, Vec<Vec<f64>>, Vec<Vector>) {
        if self.near_null_dim > 1 {
            return self.build_multi();
        }
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
            //let score = 1e-3 * almost0.norm();
            let score = almost0.norm();
            trace!("Near-Null score: {:.2e}", score);

            // Sanity check that each near null is orthogonal to the last.
            // Could move into test suite down the line.
            normalize(&mut near_null, &self.mat);
            let ortho_check: String =
                near_null_history
                    .iter()
                    .fold(String::new(), |mut acc, old| {
                        write!(
                            &mut acc,
                            "{:.1e}, ",
                            inner_product(old, &near_null, Some(&self.mat))
                        )
                        .unwrap();
                        acc
                    });

            /* could A-orthonormalize this basis but probably doesn't help
            for old in near_null_history.iter() {
                let proj = inner_product(old, &near_null, &self.mat) * old;
                near_null = near_null - proj;
                normalize(&mut near_null, &self.mat);
            }
            */

            near_null_history.push(near_null.clone());
            if !near_null_history.is_empty() {
                trace!("Near null component inner product with history: {ortho_check}");
            }

            let mut hierarchy = Hierarchy::new(self.mat.clone());
            let mut levels = 1;

            let mut block_size = self.block_size;
            loop {
                near_null /= near_null.norm();
                near_null = hierarchy.add_level(
                    &near_null,
                    self.coarsening_factor,
                    self.interpolation_type,
                    block_size,
                );
                block_size = 1;

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

                let coarse_smoother = Arc::new(L1::new(&current_a));
                //let coarse_smoother = Arc::new(SymmetricGaussSeidel::new(current_a.clone()));
                find_near_null_coarse(current_a, coarse_smoother, &mut near_null, 3);
            }
            info!("Hierarchy info: {:?}", hierarchy);
            hierarchy.print_table();

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

    pub fn build_multi(&self) -> (Composite, Vec<Vec<f64>>, Vec<Vector>) {
        let mut preconditioner = Composite::new(self.mat.clone());
        let test_data = Vec::new();
        let hist = Vec::new();
        let results_filename = "results.txt";
        let path = output_path(results_filename);
        let mut results_file = File::create(path).unwrap();
        let slurm_job_id = std::env::var("SLURM_JOB_ID");
        let slurm_job_name = std::env::var("SLURM_JOB_NAME");
        if slurm_job_id.is_ok() && slurm_job_name.is_ok() {
            results_file.write_all(
                format!(
                    "job name: {}, job ID: {}\n",
                    slurm_job_name.unwrap(),
                    slurm_job_id.unwrap()
                )
                .as_bytes(),
            );
        }

        let dim = self.mat.rows();

        // Find initial near null to get the iterations started
        //let fine_smoother = Arc::new(SymmetricGaussSeidel::new(self.mat.clone()));
        let fine_smoother = Arc::new(L1::new(&self.mat));
        let stationary = Iterative::new(self.mat.clone(), None)
            .with_solver(IterativeMethod::StationaryIteration)
            .with_max_iter(10)
            .with_preconditioner(fine_smoother.clone());
        let zeros = Vector::from(vec![0.0; dim]);
        let mut near_nullspace: Vec<Vector> = (0..self.near_null_dim)
            .map(|_| Vector::random(dim, Uniform::new(-1., 1.)))
            .collect();
        near_nullspace[0] = Vector::from_elem(dim, 1.0);
        //let ip_op = Some(self.mat.as_ref());
        let ip_op = None;
        orthonormalize_mgs(&mut near_nullspace, ip_op);
        for vec in near_nullspace.iter_mut() {
            stationary.apply_input(&zeros, vec);
        }
        orthonormalize_mgs(&mut near_nullspace, ip_op);
        for vec in near_nullspace.iter_mut() {
            stationary.apply_input(&zeros, vec);
        }
        orthonormalize_mgs(&mut near_nullspace, ip_op);

        loop {
            let mut hierarchy = Hierarchy::new(self.mat.clone());
            let mut levels = 1;

            let mut block_size = self.block_size;
            let mut current_mat = self.mat.clone();
            let mut ndofs = dim;
            let mut matrix_nullspace = Matrix::zeros((ndofs, self.near_null_dim));
            for (i, mut col) in matrix_nullspace.columns_mut().into_iter().enumerate() {
                for (a, b) in col.iter_mut().zip(near_nullspace[i].iter()) {
                    *a = *b;
                }
            }

            loop {
                let mut near_null = Vector::zeros(ndofs);
                for vec in matrix_nullspace.columns() {
                    near_null += &vec.to_owned();
                }
                near_null /= near_null.norm_l2();
                let arc_nn = Arc::new(near_null.clone());

                let mut partition_builder = PartitionBuilder::new(current_mat.clone(), arc_nn);
                let cf = self.coarsening_factor * (self.near_null_dim as f64 / block_size as f64);
                partition_builder.max_agg_size = Some((cf * 1.2).ceil() as usize);
                let mut min_agg = ((cf.ceil() + 2.) / 2.0).floor();
                let min_agg_for_sa = (self.near_null_dim as f64 / block_size as f64).ceil();
                if min_agg < min_agg_for_sa {
                    min_agg = min_agg_for_sa;
                }
                if min_agg < 1. {
                    min_agg = 1.;
                }
                partition_builder.min_agg_size = Some(min_agg as usize);
                partition_builder.coarsening_factor = cf;
                //partition_builder.max_refinement_iters = 0;

                /*
                let near_nulls: Vec<Arc<Vector>> = matrix_nullspace
                    .columns()
                    .into_iter()
                    .map(|col| Arc::new(col.to_owned()))
                    .collect();
                partition_builder.near_nulls = Some(near_nulls);
                */

                if block_size > 1 {
                    partition_builder.vector_dim = block_size;
                    partition_builder.block_reduction_strategy =
                        Some(BlockReductionStrategy::default());
                }
                let partition = partition_builder.build();

                let (coarse_near_nullspace, r, p, mat_coarse) =
                    smoothed_aggregation2(&current_mat, &partition, block_size, &matrix_nullspace);
                current_mat = Arc::new(mat_coarse);

                hierarchy.push_level(
                    current_mat.clone(),
                    Arc::new(r),
                    Arc::new(p),
                    Arc::new(partition),
                    Arc::new(matrix_nullspace),
                    Arc::new(near_null),
                    block_size,
                );

                ndofs = current_mat.rows();
                matrix_nullspace = coarse_near_nullspace;

                block_size = self.near_null_dim;

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
                    < 100 * self.near_null_dim
                //< 2000
                {
                    break;
                }

                let coarse_smoother = Arc::new(L1::new(&current_mat));
                let stationary = Iterative::new(current_mat.clone(), None)
                    .with_solver(IterativeMethod::StationaryIteration)
                    .with_max_iter(5)
                    .with_preconditioner(coarse_smoother);
                let zeros = Vector::zeros(ndofs);

                for mut col in matrix_nullspace.columns_mut() {
                    let mut col_own = col.to_owned();
                    stationary.apply_input(&zeros, &mut col_own);
                    col_own /= col_own.norm_l2();
                    col.iter_mut()
                        .zip(col_own.iter())
                        .for_each(|(a, b)| *a = *b);
                }
            }
            info!("Hierarchy info: {:?}", hierarchy);
            // TODO this shoud be in Debug impl along with interp::info...
            hierarchy.print_table();

            let filename = format!("hierarchy_{}.json", preconditioner.components().len());
            let path = output_path(filename);
            let data: HierarchyData = hierarchy.clone().into();
            let mut file = File::create(path).unwrap();
            let serialized = serde_json::to_string(&data).unwrap();
            file.write_all(&serialized.as_bytes()).unwrap();

            let complexity = hierarchy.op_complexity();
            let ml1 = Arc::new(Multilevel::new(
                hierarchy,
                self.solve_coarsest_exactly,
                self.smoother_type,
                self.smoothing_steps,
                self.mu,
            ));
            preconditioner.push(ml1);

            near_nullspace = (0..self.near_null_dim)
                .map(|_| Vector::random(dim, Uniform::new(-1., 1.)))
                .collect();
            let (convergence_rate, convergence_history) = find_near_null_multi(
                self.mat.clone(),
                &preconditioner,
                &mut near_nullspace,
                self.test_iters,
            );
            let results_string: String = format!(
                "{}: {:.3}, {:.3}\n",
                preconditioner.components().len(),
                complexity,
                convergence_rate
            );
            results_file.write_all(results_string.as_bytes()).unwrap();

            if let Some(max_components) = self.max_components {
                if preconditioner.components().len() >= max_components {
                    return (preconditioner, test_data, hist);
                }
            }
            if let Some(target_convergence) = self.target_convergence {
                if convergence_rate < target_convergence {
                    return (preconditioner, test_data, hist);
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
        *near_null /= near_null.norm();
        let a_norm = norm(near_null, &mat);
        let stationary = Iterative::new(mat.clone(), Some(near_null.clone()))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Arc::new(composite_preconditioner.clone()))
            .with_max_iter(1);
        *near_null = stationary.apply(&zeros);
        iter += 1;
        let convergence_factor = norm(near_null, &mat) / a_norm;
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

fn find_near_null_multi(
    mat: Arc<CsrMatrix>,
    composite_preconditioner: &Composite,
    near_nullspace: &mut Vec<Vector>,
    test_iters: Option<usize>,
) -> (f64, Vec<f64>) {
    let mut iter = 0;
    let max_iter = test_iters.unwrap_or(usize::MAX);
    let max_time = Duration::from_secs(3000);
    let start = Instant::now();
    let mut last_log = start;
    let log_interval = Duration::from_secs(10);
    //let zeros = Vector::from(vec![0.0; mat.rows()]);
    let cycles = ((composite_preconditioner.components().len() * 2) - 1) as f64;
    let mut old_convergence_factor = 0.0;
    let mut history = Vec::new();
    //let ip_op = Some(mat.as_ref());
    let ip_op = None;
    orthonormalize_mgs(near_nullspace, ip_op);
    let mut near_null = Vector::from(vec![0.0; mat.rows()]);
    for vec in near_nullspace.iter() {
        near_null = near_null + vec;
    }
    normalize(&mut near_null, &mat);

    loop {
        /*
        let stationary = Iterative::new(mat.clone(), None)
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Arc::new(composite_preconditioner.clone()))
            .with_max_iter(1);
        */
        for vec in near_nullspace.iter_mut() {
            //stationary.apply_input(&zeros, vec);
            let ax = spmv(&mat, vec);
            let c = &composite_preconditioner.apply(&ax);
            *vec = &*vec - c;
        }
        //iter += 1;
        iter += composite_preconditioner.components().len() * 2 - 1;
        /*
        let mut avg_convergence = 0.0;
        for near_null in near_nullspace.iter() {
            let convergence_factor = norm(near_null, &mat);
            //let convergence_factor = near_null.norm_l2();
            if convergence_factor > 1.0 {
                if ip_op.is_some() {
                    error!(
                        "Not convergent method! Tester convergence factor: {:.2} on iteration {}",
                        convergence_factor, iter
                    );
                //maybe panic?
                } else {
                    warn!(
                    "Not convergent in L2 norm! This is possible for a convergent operator as long as the tester convergence is close it 1.0, Tester convergence factor: {:.2} on iteration {}",
                    convergence_factor, iter
                    );
                }
            }
            avg_convergence += convergence_factor;
        }
        avg_convergence /= near_nullspace.len() as f64;
        // should only be monotonic if we monitor the convergence factor in the A-norm
        if ip_op.is_some() && avg_convergence < old_convergence_factor {
            warn!("Monotonicity properties violated in tester at iter: {}, cf_i: {:.2}, cf_i-1: {:.2}", iter, avg_convergence, old_convergence_factor);
        }
        history.push(avg_convergence);
        */
        let ax = spmv(&mat, &near_null);
        let c = &composite_preconditioner.apply(&ax);
        near_null = near_null - c;
        let convergence_factor = norm(&near_null, &mat);
        if convergence_factor > 1.0 {
            error!(
                "Not convergent method! Tester convergence factor: {:.2} on iteration {}",
                convergence_factor, iter
            );
        }
        if convergence_factor < old_convergence_factor {
            warn!("Monotonicity properties violated in tester at iter: {}, cf_i: {:.2}, cf_i-1: {:.2}", iter, convergence_factor, old_convergence_factor);
        }
        history.push(convergence_factor);

        orthonormalize_mgs(near_nullspace, ip_op);
        near_null = Vector::from(vec![0.0; mat.rows()]);
        for vec in near_nullspace.iter() {
            near_null = near_null + vec;
        }
        normalize(&mut near_null, &mat);

        let now = Instant::now();
        let elapsed = now - start;
        let elapsed_secs = elapsed.as_millis() as f64 / 1000.0;

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
        if iter >= max_iter || elapsed > max_time {
            info!(
                "{} components:\n\tconvergence factor: {:.3}\n\tconvergence factor per cycle: {:.3}\n\tsearch iters: {}\n\tsearch time: {}",
                composite_preconditioner.components().len(),
                convergence_factor,
                convergence_factor.powf(1.0 / cycles),
                iter,
                format_duration(&elapsed)
            );
            return (convergence_factor.powf(1.0 / cycles), history);
        }
        old_convergence_factor = convergence_factor;
    }
}
