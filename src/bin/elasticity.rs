use std::sync::Arc;
use std::{fs::File, io::Write, time::Duration};

use amg_match::interpolation::InterpolationType;
use amg_match::preconditioner::{LinearOperator, Multilevel, SmootherType, L1};
use amg_match::{
    adaptive::AdaptiveBuilder,
    preconditioner::Composite,
    solver::{Iterative, IterativeMethod, LogInterval, SolveInfo},
    utils::{format_duration, load_system},
};
use amg_match::{CooMatrix, CsrMatrix, Vector};
use chrono::format::format;
use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
use ndarray::{Array, Array6};
use ndarray_linalg::krylov::{AppendResult, Orthogonalizer, MGS};
use ndarray_linalg::{InnerProduct, Norm, SVDInto};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use sprs::io::read_matrix_market;

#[macro_use]
extern crate log;

fn main() {
    pretty_env_logger::init();

    let prefix = "data/elasticity";
    let name = "elasticity_3d";

    let (mat, b, _coords, rbms, truedofs_map) = load_system(prefix, name, false);
    let nrows = b.len();
    let rbm_smoothing_steps = 3;
    let rbms = smooth_rbms(
        rbms.unwrap(),
        mat.clone(),
        &truedofs_map,
        rbm_smoothing_steps,
    );
    //value_aggregation(mat, b, &truedofs_map);
    let pc = build_pc(mat, name);
    eval_nearnull_and_rbm_spaces(rbms, Arc::new(pc));
}

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

    rbms.into_iter()
        .map(|rbm| {
            let mut free = truedofs_map * &rbm;
            smoother.apply_input(&zeros, &mut free);
            free /= free.norm();
            free
        })
        .collect()
}

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
