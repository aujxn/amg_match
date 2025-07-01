use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use amg_match::{
    parallel_ops::spmv,
    preconditioner::BlockSmootherType,
    solver::{Iterative, IterativeMethod, LogInterval},
    utils::{format_duration, load_system, norm, normalize},
};

use amg_match::hierarchy::Hierarchy;
use amg_match::interpolation::smoothed_aggregation2;
use amg_match::partitioner::{BlockReductionStrategy, PartitionBuilder};
use amg_match::preconditioner::{
    BlockSmoother, LinearOperator, Multilevel, SymmetricGaussSeidel, L1,
};
use amg_match::solver::Direct;
use amg_match::utils::orthonormalize_mgs;
use amg_match::{CsrMatrix, Vector};
use ndarray::{stack, ArrayView, Axis};
use ndarray_linalg::Norm;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[macro_use]
extern crate log;

fn main() {
    pretty_env_logger::init();

    let search_iter = 200;
    let n_candidates = 1;
    let mut results = Vec::new();

    for refine in 4..8 {
        let prefix = "data/laplace";
        let name = format!("{}", refine);
        let (mat, b, _coords, _rbms, _truedofs_map) = load_system(&prefix, &name, false);
        let dim = mat.rows();
        info!("Starting: {} with {} dofs", refine, dim);
        let mut near_nullspace = Vec::new();
        for _ in 0..n_candidates {
            let near_null = Vector::random(dim, Uniform::new(-1., 1.));
            near_nullspace.push(near_null);
        }
        let test_pc = Arc::new(L1::new(&mat));

        let (_convergence_factor, _hist) =
            find_near_null_multi(mat.clone(), test_pc, &mut near_nullspace, search_iter);

        let result = sa_test(mat, &b, &near_nullspace);
        results.push((refine, result));

        println!(
            "{:>8}  {:>8}  {:>13}  {:>17}  {:>12}  {:>17}  {:>12}",
            "refine",
            "ndofs",
            "op complexity",
            "stationary gs",
            "cg gs",
            "stationary ilu",
            "cg ilu"
        );

        println!(
            "{:-<8}  {:-<8}  {:-<13}  {:-<17}  {:-<12}  {:-<17}  {:-<12}",
            "", "", "", "", "", "", "",
        );

        for (refine, result) in results.iter() {
            println!(
                "{:>8}  {:>8}  {:>13.2}  {:>17}  {:>12}  {:>17}  {:>12}",
                refine, result.0, result.1, result.2, result.3, result.4, result.5
            );
        }
    }
}

fn find_near_null_multi(
    mat: Arc<CsrMatrix>,
    pc: Arc<dyn LinearOperator>,
    near_nullspace: &mut Vec<Vector>,
    max_iter: usize,
) -> (f64, Vec<f64>) {
    let mut iter = 0;
    let start = Instant::now();
    let mut last_log = start;
    let log_interval = Duration::from_secs(10);
    //let zeros = Vector::from(vec![0.0; mat.rows()]);
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
        for vec in near_nullspace.iter_mut() {
            let ax = spmv(&mat, vec);
            let c = &pc.apply(&ax);
            *vec = &*vec - c;
        }
        iter += 1;

        let ax = spmv(&mat, &near_null);
        let c = &pc.apply(&ax);
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
                "iteration {}:\n\ttotal search time: {:.0}s\n\tConvergence Factor: {:.3}",
                iter,
                elapsed_secs,
                convergence_factor,
            );
            last_log = now;
        }

        //if old_convergence_factor / convergence_factor > 0.999 || iter >= max_iter {
        if iter >= max_iter {
            info!(
                "convergence factor: {:.3}\n\tsearch iters: {}\n\tsearch time: {}",
                convergence_factor,
                iter,
                format_duration(&elapsed)
            );
            return (convergence_factor, history);
        }
        old_convergence_factor = convergence_factor;
    }
}

fn build_smoothers(
    mat: Arc<CsrMatrix>,
    block_size: usize,
    near_null: Arc<Vector>,
    vdim: usize,
) -> (
    Arc<dyn LinearOperator + Sync + Send>,
    Arc<dyn LinearOperator + Sync + Send>,
) {
    let cf;
    if vdim * mat.rows() / block_size < 32 {
        cf = mat.rows() as f64 / (32.0 * vdim as f64);
    } else {
        cf = block_size as f64 / vdim as f64;
    }
    info!(
        "Building smoother: {:.2} cf, {} vdim, and {} len nn",
        cf,
        vdim,
        near_null.len()
    );
    let mut builder = PartitionBuilder::new(mat.clone(), near_null);
    builder.coarsening_factor = cf;
    let max_agg = (cf * 1.1).ceil();
    let min_agg = (max_agg / 1.5).floor();
    builder.max_agg_size = Some(max_agg as usize);
    builder.min_agg_size = Some(min_agg as usize);
    builder.vector_dim = vdim;
    builder.block_reduction_strategy = Some(BlockReductionStrategy::default());
    let partition = Arc::new(builder.build());

    partition.info();
    let smoother_gs = Arc::new(BlockSmoother::new(
        &*mat,
        partition.clone(),
        BlockSmootherType::GaussSeidel,
        vdim,
    ));
    let smoother_ilu = Arc::new(BlockSmoother::new(
        &*mat,
        partition,
        BlockSmootherType::IncompleteCholesky,
        vdim,
    ));

    let zeros = Vector::from(vec![0.0; mat.cols()]);
    let forward_solver_gs = Iterative::new(mat.clone(), Some(zeros.clone()))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_max_iter(1)
        .with_preconditioner(smoother_gs)
        .with_relative_tolerance(1e-8)
        .with_absolute_tolerance(f64::EPSILON);

    let forward_solver_ilu = Iterative::new(mat.clone(), Some(zeros.clone()))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_max_iter(1)
        .with_preconditioner(smoother_ilu)
        .with_relative_tolerance(1e-8)
        .with_absolute_tolerance(f64::EPSILON);
    (Arc::new(forward_solver_gs), Arc::new(forward_solver_ilu))
}

fn sa_test(
    mat: Arc<CsrMatrix>,
    b: &Vector,
    near_nullspace: &Vec<Vector>,
) -> (usize, f64, usize, usize, usize, usize) {
    let coarsening_factor = 8.0;
    let smoothing_steps = 3;
    let near_null_dim = near_nullspace.len();
    let mut block_size = 1;
    let mut ndofs = b.len();
    let mut current_mat = mat.clone();

    let mut hierarchy = Hierarchy::new(mat.clone());
    let mut near_null = Vector::zeros(ndofs);
    for vec in near_nullspace.iter() {
        near_null += vec;
    }
    let near_nullspace: Vec<ArrayView<_, _>> =
        near_nullspace.iter().map(|col| col.view()).collect();
    let mut near_nullspace = stack(Axis(1), &near_nullspace).unwrap();

    while ndofs > 100 {
        //let smoother = L1::new(&current_mat);
        let smoother = SymmetricGaussSeidel::new(current_mat.clone());
        let zeros = Vector::from_elem(ndofs, 0.0);
        let smoother = Iterative::new(current_mat.clone(), Some(zeros.clone()))
            .with_max_iter(smoothing_steps)
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Arc::new(smoother));

        for mut col in near_nullspace.columns_mut() {
            let mut col_own = col.to_owned();
            smoother.apply_input(&zeros, &mut col_own);
            col_own /= col_own.norm_l2();
            col.iter_mut()
                .zip(col_own.iter())
                .for_each(|(a, b)| *a = *b);
        }
        //let ip_op = None;
        //orthonormalize_mgs(&mut near_nullspace, ip_op);
        near_null = near_nullspace.sum_axis(Axis(1));
        near_null /= near_null.norm_l2();

        let arc_nn = Arc::new(near_null.clone());
        let mut partition_builder = PartitionBuilder::new(current_mat.clone(), arc_nn);
        let cf = coarsening_factor * (near_null_dim as f64 / block_size as f64);
        partition_builder.max_agg_size = Some((cf * 1.5).ceil() as usize);
        let mut min_agg = ((cf.ceil() + 2.) / 2.0).floor();
        let min_agg_for_sa = (near_null_dim as f64 / block_size as f64).ceil();
        if min_agg < min_agg_for_sa {
            min_agg = min_agg_for_sa;
        }
        if min_agg < 1. {
            min_agg = 1.;
        }
        partition_builder.min_agg_size = Some(min_agg as usize);
        partition_builder.coarsening_factor = cf;
        if block_size > 1 {
            partition_builder.vector_dim = block_size;
            partition_builder.block_reduction_strategy = Some(BlockReductionStrategy::default());
        }
        let partition = partition_builder.build();

        let (coarse_near_nullspace, r, p, mat_coarse) =
            smoothed_aggregation2(&current_mat, &partition, block_size, &near_nullspace);
        current_mat = Arc::new(mat_coarse);

        hierarchy.push_level(
            current_mat.clone(),
            Arc::new(r),
            Arc::new(p),
            Arc::new(partition),
            Arc::new(near_null),
            block_size,
        );

        ndofs = current_mat.rows();

        near_nullspace = coarse_near_nullspace;

        block_size = near_null_dim;
    }

    info!("Hierarchy info: {:?}", hierarchy);
    hierarchy.print_table();
    println!("{:?}", hierarchy.vdims);

    let mut smoothers_gs = Vec::new();
    let mut smoothers_ilu = Vec::new();

    let compensated_block_size = 500;

    let (fine_smoother_gs, fine_smoother_ilu) = build_smoothers(
        mat.clone(),
        compensated_block_size,
        hierarchy.get_near_null(0).clone(),
        hierarchy.vdims[0],
    );

    smoothers_gs.push(fine_smoother_gs);
    smoothers_ilu.push(fine_smoother_ilu);

    let coarse_index = hierarchy.get_coarse_mats().len() - 1;
    for (level, mat) in hierarchy.get_coarse_mats().iter().enumerate() {
        if level == coarse_index {
            let solver = Arc::new(Direct::new(&mat));
            smoothers_gs.push(solver.clone());
            smoothers_ilu.push(solver);
        } else {
            let (gs, ilu) = build_smoothers(
                mat.clone(),
                compensated_block_size,
                hierarchy.get_near_null(level + 1).clone(),
                hierarchy.vdims[level + 1],
            );

            smoothers_gs.push(gs);
            smoothers_ilu.push(ilu);
        }
    }

    let ml_gs = Arc::new(Multilevel {
        hierarchy: hierarchy.clone(),
        forward_smoothers: smoothers_gs,
        backward_smoothers: None,
        mu: 1,
    });
    let ml_ilu = Arc::new(Multilevel {
        hierarchy: hierarchy.clone(),
        forward_smoothers: smoothers_ilu,
        backward_smoothers: None,
        mu: 1,
    });

    let op_comp = ml_gs.get_hierarchy().op_complexity();
    let epsilon = 1e-12;
    let max_iter = 1000;
    let guess = Vector::random(mat.rows(), Uniform::new(-1., 1.));
    let mut solver = Iterative::new(mat.clone(), Some(guess))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_preconditioner(ml_gs)
        .with_relative_tolerance(epsilon)
        .with_log_interval(LogInterval::Iterations(10))
        .with_max_iter(max_iter);

    let (_, solve_info) = solver.solve(b);
    let stationary_gs = solve_info.iterations;

    solver = solver.with_solver(IterativeMethod::ConjugateGradient);
    let (_, solve_info) = solver.solve(b);
    let cg_gs = solve_info.iterations;

    solver = solver.with_preconditioner(ml_ilu);
    solver = solver.with_solver(IterativeMethod::StationaryIteration);
    let (_, solve_info) = solver.solve(b);
    let stationary_ilu = solve_info.iterations;

    solver = solver.with_solver(IterativeMethod::ConjugateGradient);
    let (_, solve_info) = solver.solve(b);
    let cg_ilu = solve_info.iterations;

    (
        b.len(),
        op_comp,
        stationary_gs,
        cg_gs,
        stationary_ilu,
        cg_ilu,
    )
}
