use std::sync::Arc;

use amg_match::hierarchy::Hierarchy;
use amg_match::interpolation::smoothed_aggregation2;
use amg_match::partitioner::{BlockReductionStrategy, PartitionBuilder};
use amg_match::preconditioner::{
    BlockSmoother, BlockSmootherType, LinearOperator, Multilevel, SmootherType,
    SymmetricGaussSeidel, L1,
};
use amg_match::solver::Direct;
use amg_match::utils::orthonormalize_mgs;
use amg_match::{
    solver::{Iterative, IterativeMethod, LogInterval},
    utils::load_system,
};
use amg_match::{CsrMatrix, Vector};
use ndarray::{stack, Axis};
use ndarray_linalg::Norm;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[macro_use]
extern crate log;

fn main() {
    pretty_env_logger::init();

    let mut results = Vec::new();
    for refine in 2..7 {
        let prefix = format!("data/elasticity/{}", refine);
        let name = "elasticity_3d";

        let (mat, b, _coords, rbms, truedofs_map) = load_system(&prefix, name, false);
        let rbm_smoothing_steps = 10;
        let rbms = smooth_rbms(
            rbms.unwrap(),
            mat.clone(),
            &truedofs_map,
            rbm_smoothing_steps,
        );
        //value_aggregation(mat, b, &truedofs_map);
        //let pc = build_pc(mat, name);
        //eval_nearnull_and_rbm_spaces(rbms, Arc::new(pc));

        let result = sa_test(mat, &b, &rbms);
        results.push((refine, result));

        println!(
            "{:>8}  {:>8}  {:>13}  {:>17}  {:>12}  {:>17}  {:>12}",
            "refine",
            "ndofs",
            "op complexity",
            "stationary block",
            "cg block",
            "stationary scalar",
            "cg scalar"
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

fn sa_test(
    mat: Arc<CsrMatrix>,
    b: &Vector,
    smooth_rbms: &Vec<Vector>,
) -> (usize, f64, usize, usize, usize, usize) {
    let coarsening_factor = 12.0;
    let smoothing_steps = 10;
    let mut block_size = 3;
    let mut ndofs = b.len();
    let mut current_mat = mat.clone();

    let mut hierarchy = Hierarchy::new(mat.clone());
    let mut near_null = Vector::zeros(ndofs);
    for rbm in smooth_rbms.iter() {
        near_null += rbm;
    }
    let mut near_nullspace = stack![
        Axis(1),
        smooth_rbms[0],
        smooth_rbms[1],
        smooth_rbms[2],
        smooth_rbms[3],
        smooth_rbms[4],
        smooth_rbms[5],
    ];

    while ndofs > 100 {
        //let l1 = L1::new(&current_mat);
        let l1 = SymmetricGaussSeidel::new(current_mat.clone());
        let zeros = Vector::from_elem(ndofs, 0.0);
        let smoother = Iterative::new(current_mat.clone(), Some(zeros.clone()))
            .with_max_iter(smoothing_steps)
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Arc::new(l1));

        //smoother.apply_input(&zeros, &mut near_null);
        //let mut basis = Vec::new();
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
        let near_null_dim = 6;
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

        block_size = 6;
    }

    info!("Hierarchy info: {:?}", hierarchy);
    hierarchy.print_table();
    println!("{:?}", hierarchy.vdims);

    let mut hierarchy_scalar = hierarchy.clone();
    for val in hierarchy_scalar.vdims.iter_mut() {
        *val = 1;
    }
    let mut smoothers = Vec::new();
    let mut smoothers_scalar = Vec::new();

    let compensated_block_size = 500;
    let block_smoother_type = BlockSmootherType::GaussSeidel;
    //BlockSmootherType::AutoCholesky(sprs::FillInReduction::CAMDSuiteSparse);

    let (fine_smoother_block, fine_smoother_scalar) = build_smoothers(
        mat.clone(),
        block_smoother_type,
        compensated_block_size,
        hierarchy.get_near_null(0).clone(),
        hierarchy.vdims[0],
    );

    smoothers.push(fine_smoother_block);
    smoothers_scalar.push(fine_smoother_scalar);

    let coarse_index = hierarchy.get_coarse_mats().len() - 1;
    for (level, mat) in hierarchy.get_coarse_mats().iter().enumerate() {
        if level == coarse_index {
            let solver = Arc::new(Direct::new(&mat));
            smoothers.push(solver.clone());
            smoothers_scalar.push(solver);
        } else {
            let (block, scalar) = build_smoothers(
                mat.clone(),
                block_smoother_type,
                compensated_block_size,
                hierarchy.get_near_null(level + 1).clone(),
                hierarchy.vdims[level + 1],
            );

            smoothers.push(block);
            smoothers_scalar.push(scalar);
        }
    }

    let ml = Arc::new(Multilevel {
        hierarchy: hierarchy.clone(),
        forward_smoothers: smoothers,
        backward_smoothers: None,
        mu: 1,
    });
    let ml_scalar = Arc::new(Multilevel {
        hierarchy: hierarchy.clone(),
        forward_smoothers: smoothers_scalar,
        backward_smoothers: None,
        mu: 1,
    });

    let op_comp = ml.get_hierarchy().op_complexity();
    let epsilon = 1e-12;
    let max_iter = 1000;
    let guess = Vector::random(mat.rows(), Uniform::new(-1., 1.));
    let mut solver = Iterative::new(mat.clone(), Some(guess))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_preconditioner(ml)
        .with_relative_tolerance(epsilon)
        .with_log_interval(LogInterval::Iterations(10))
        .with_max_iter(max_iter);

    let (_, solve_info) = solver.solve(b);
    let stationary_block = solve_info.iterations;

    solver = solver.with_solver(IterativeMethod::ConjugateGradient);
    let (_, solve_info) = solver.solve(b);
    let cg_block = solve_info.iterations;

    solver = solver.with_preconditioner(ml_scalar);
    solver = solver.with_solver(IterativeMethod::StationaryIteration);
    let (_, solve_info) = solver.solve(b);
    let stationary_scalar = solve_info.iterations;

    solver = solver.with_solver(IterativeMethod::ConjugateGradient);
    let (_, solve_info) = solver.solve(b);
    let cg_scalar = solve_info.iterations;

    (
        b.len(),
        op_comp,
        stationary_block,
        cg_block,
        stationary_scalar,
        cg_scalar,
    )
}

fn build_smoothers(
    mat: Arc<CsrMatrix>,
    smoother: BlockSmootherType,
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
    let smoother_block = Arc::new(BlockSmoother::new(&*mat, partition.clone(), smoother, vdim));
    let smoother_scalar = Arc::new(BlockSmoother::new(&*mat, partition, smoother, 1));

    let zeros = Vector::from(vec![0.0; mat.cols()]);
    let forward_solver = Iterative::new(mat.clone(), Some(zeros.clone()))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_max_iter(1)
        .with_preconditioner(smoother_block)
        .with_relative_tolerance(1e-8)
        .with_absolute_tolerance(f64::EPSILON);

    let forward_solver_scalar = Iterative::new(mat.clone(), Some(zeros.clone()))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_max_iter(1)
        .with_preconditioner(smoother_scalar)
        .with_relative_tolerance(1e-8)
        .with_absolute_tolerance(f64::EPSILON);
    (Arc::new(forward_solver), Arc::new(forward_solver_scalar))
}

/*
fn value_aggregation(mat: Arc<CsrMatrix>, b: Vector, truedofs_map: &CsrMatrix) {
    let truedof_idx = truedofs_map.indices();
    let nrows = mat.rows();

    let mut big_block_coo = CooMatrix::new((nrows, nrows));
    for (i, row) in mat.outer_iterator().enumerate() {
        for (j, val) in row.iter() {
            if *val != 0.0 {
                let block_i = truedof_idx[i] % 3;
                let block_j = truedof_idx[j] % 3;
                if block_i == block_j {
                    big_block_coo.add_triplet(i, j, *val);
                }
            }
        }
    }
    let block_csr: Arc<CsrMatrix> = Arc::new(big_block_coo.to_csr());

    let adaptive_builder = AdaptiveBuilder::new(block_csr)
        .with_max_components(1)
        .with_coarsening_factor(7.5)
        .with_smoother(SmootherType::BlockGaussSeidel)
        .with_interpolator(InterpolationType::SmoothedAggregation((1, 0.66)))
        .with_smoothing_steps(1)
        .cycle_type(1)
        .with_max_test_iters(50);
    let (pc, _convergence_hist, _near_nulls) = adaptive_builder.build();

    let mut hierarchy = pc.components()[0].hierarchy.clone();
    hierarchy.set_fine_mat(mat.clone());
    info!("Hierarchy info: {:?}", hierarchy);
    let pc = Arc::new(Multilevel::new(
        hierarchy,
        true,
        SmootherType::BlockGaussSeidel,
        1,
        1,
    ));

    let epsilon = 1e-12;
    let stationary = Iterative::new(mat.clone(), None)
        .with_relative_tolerance(epsilon)
        .with_solver(IterativeMethod::StationaryIteration)
        .with_preconditioner(pc)
        .with_log_interval(LogInterval::Iterations(500));
    stationary.solve(&b);

    let cg = stationary
        .with_solver(IterativeMethod::ConjugateGradient)
        .with_log_interval(LogInterval::Iterations(10));
    cg.solve(&b);
}

fn build_pc(mat: Arc<CsrMatrix>, name: &str) -> Composite {
    info!("nrows: {} nnz: {}", mat.rows(), mat.nnz());
    let max_components = 6;
    let coarsening_factor = 7.5;
    let test_iters = 15;

    let adaptive_builder = AdaptiveBuilder::new(mat.clone())
        .with_max_components(max_components)
        .with_coarsening_factor(coarsening_factor)
        //.with_smoother(SmootherType::BlockL1)
        .with_smoother(SmootherType::BlockGaussSeidel)
        .with_interpolator(InterpolationType::SmoothedAggregation((1, 0.66)))
        .with_smoothing_steps(1)
        .cycle_type(1)
        .with_max_test_iters(test_iters);

    info!("Starting {} CF-{:.0}", name, coarsening_factor);
    let timer = std::time::Instant::now();
    let (pc, _convergence_hist, _near_nulls) = adaptive_builder.build();

    let construction_time = timer.elapsed();
    info!(
        "Preconitioner built in: {}",
        format_duration(&construction_time)
    );

    pc
}
*/

fn smooth_rbms(
    rbms: Vec<Vector>,
    mat: Arc<CsrMatrix>,
    truedofs_map: &CsrMatrix,
    smoothing_steps: usize,
) -> Vec<Vector> {
    let l1 = L1::new(&mat);
    let nrows = mat.rows();
    let zeros = Vector::from_elem(nrows, 0.0);
    let smoother = Iterative::new(mat.clone(), Some(zeros.clone()))
        .with_max_iter(smoothing_steps)
        .with_solver(IterativeMethod::StationaryIteration)
        .with_preconditioner(Arc::new(l1));

    let mut smooth_rbms: Vec<Vector> = rbms
        .into_iter()
        .map(|rbm| {
            /*
            let mut free = truedofs_map * &rbm;
            smoother.apply_input(&zeros, &mut free);
            free /= free.norm();
            free
            */
            truedofs_map * &rbm
        })
        .collect();
    let ip_op = None;
    orthonormalize_mgs(&mut smooth_rbms, ip_op);
    for vec in smooth_rbms.iter_mut() {
        smoother.apply_input(&zeros, vec);
    }
    orthonormalize_mgs(&mut smooth_rbms, ip_op);
    smooth_rbms
}

/*
fn eval_nearnull_and_rbm_spaces(rbms: Vec<Vector>, pc: Arc<Composite>) {
    let nrows = pc.get_mat().rows();

    let mut mgs_rbm = MGS::new(nrows, 1e-12);
    for vec in rbms.iter() {
        match mgs_rbm.append(vec.clone()) {
            AppendResult::Added(_) => (),
            AppendResult::Dependent(_) => {
                error!("rbms are dependent...");
                panic!()
            }
        }
    }
    let rbm_q = mgs_rbm.get_q();
    let rbm_qt = rbm_q.t();

    let mut mgs_nearnull = MGS::new(nrows, 1e-12);
    for vec in pc
        .components()
        .iter()
        .map(|comp| comp.get_hierarchy().get_near_null(0))
    {
        let vec: &Vector = vec;
        match mgs_nearnull.append(vec.clone()) {
            AppendResult::Added(_) => (),
            AppendResult::Dependent(_) => {
                error!("near_nulls are dependent...");
                panic!()
            }
        }
    }
    let near_null_q = mgs_nearnull.get_q();
    let mut c = Array::zeros((rbm_qt.nrows(), near_null_q.ncols()));
    general_mat_mul(1.0, &rbm_qt, &near_null_q, 0.0, &mut c);
    let (_u, s, _vt) = c.svd_into(false, false).unwrap();
    let svds_string: String = s.iter().map(|sval| format!("{:5.2} ", sval)).collect();
    trace!("SVDs of Q_rbm_t Q_nn: {}", svds_string);

    let score = s.iter().sum::<f64>() / rbms.len() as f64;
    trace!("(1/{})*||Q_rbm* Q_nn||_1: = {:.3}", rbms.len(), score);

    let nn_qt = near_null_q.t();
    let mut scores = String::from("");
    for rbm in rbms.iter() {
        let mut coefs = Vector::zeros(nn_qt.nrows());
        general_mat_vec_mul(1.0, &nn_qt, &rbm, 0.0, &mut coefs);
        scores = format!("{}{:6.3} ", scores, coefs.norm());
    }
    let rbm_names = ["rxy", "ryz", "rzx", "tx", "ty", "tz"];
    let header: String = rbm_names
        .iter()
        .map(|name| format!("{:6} ", name))
        .collect();
    trace!(
        "RBMs projected onto near-null space norm ||Q_nn^T rbm||_2:\n{}\n{}",
        header,
        scores
    );
}
*/
