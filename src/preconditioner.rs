//! Definition of the `LinearOperator` trait as well as implementors
//! of said trait.
//!

use crate::hierarchy::Hierarchy;
use crate::parallel_ops::spmv;
use crate::partitioner::{metis_n, Partition};
use crate::solver::{Direct, Iterative, IterativeMethod};
use crate::{Cholesky, CooMatrix, CsrMatrix, Vector};
use ndarray::par_azip;
use ndarray_linalg::{FactorizeC, Norm, SolveC, UPLO};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use sprs::linalg::trisolve::{lsolve_csr_dense_rhs as lsolve, usolve_csr_dense_rhs as usolve};
use sprs_ldl::Ldl;
use std::sync::Arc;

/// A Trait for objects which can map from R^n to R^m that are linear. In linear algebra
/// applications, this is most things (matrices, preconditioners, solvers, etc.).
pub trait LinearOperator {
    /* Should LinearOperator have this information? On one hand, this might be nice to reduce the
     * number of arguments/returns for functions which consume and produce LinearOperators.
     * Additionally, run time verification of dimension checking could be useful for debugging.
     * On the other hand, requiring explicit size makes it way more difficult to implement this
     * trait for certain objects, namely abstract function types... Could also return an optional
     * when it is easy to know the size but that doesn't seem very nice...
     * (see impl for `F: Fn(&Vector) -> Vector`)
     */
    //fn rows(&self) -> usize;
    //fn cols(&self) -> usize;

    fn apply_mut(&self, vec: &mut Vector);

    fn apply(&self, vec: &Vector) -> Vector {
        let mut out_vec = vec.clone();
        self.apply_mut(&mut out_vec);
        out_vec
    }

    fn apply_input(&self, in_vec: &Vector, out_vec: &mut Vector) {
        out_vec.clone_from(in_vec);
        self.apply_mut(out_vec);
    }
}

impl<F> LinearOperator for F
where
    F: Fn(&Vector) -> Vector,
{
    fn apply(&self, vec: &Vector) -> Vector {
        self(vec)
    }

    fn apply_mut(&self, vec: &mut Vector) {
        let out = self(vec);
        vec.clone_from(&out);
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum SmootherType {
    L1,
    GaussSeidel,
    BlockL1,
    BlockGaussSeidel,
}

pub fn build_smoother(
    mat: Arc<CsrMatrix>,
    smoother: SmootherType,
    partition: Arc<Partition>,
    transpose: bool,
) -> Arc<dyn LinearOperator + Send + Sync> {
    match smoother {
        SmootherType::L1 => Arc::new(L1::new(&mat)),
        //SmootherType::BlockL1 => Arc::new(BlockL1Iterative::new(&mat, restriction.clone())),
        SmootherType::BlockL1 => Arc::new(BlockL1::new(&mat, partition.clone())),
        SmootherType::GaussSeidel => {
            if transpose {
                Arc::new(BackwardGaussSeidel::new(mat))
            } else {
                Arc::new(ForwardGaussSeidel::new(mat))
            }
        }
        SmootherType::BlockGaussSeidel => {
            let block_gs = BlockGaussSeidel::new(&mat, partition.clone());
            Arc::new(block_gs)
            /*
            if transpose {
                let mut block_gs = BlockGaussSeidel::new(&mat, restriction.clone());
                block_gs.set_forward(false);
                Arc::new(block_gs)
            } else {
                let block_gs = BlockGaussSeidel::new(&mat, restriction.clone());
                Arc::new(block_gs)
            }
            */
        }
    }
}

pub struct Identity;
impl Identity {
    pub fn new() -> Self {
        Self
    }
}
//unsafe impl Send for Identity {}
//unsafe impl Sync for Identity {}

impl LinearOperator for Identity {
    fn apply_mut(&self, _vec: &mut Vector) {}

    fn apply(&self, vec: &Vector) -> Vector {
        vec.clone()
    }

    fn apply_input(&self, in_vec: &Vector, out_vec: &mut Vector) {
        out_vec.clone_from(in_vec);
    }
}

pub struct L1 {
    l1_inverse: Vector,
}

impl LinearOperator for L1 {
    fn apply_mut(&self, vec: &mut Vector) {
        par_azip!((a in vec, b in &self.l1_inverse) *a *= b)
    }

    fn apply(&self, vec: &Vector) -> Vector {
        vec * &self.l1_inverse
    }

    fn apply_input(&self, in_vec: &Vector, out_vec: &mut Vector) {
        par_azip!((a in out_vec, &b in in_vec, &c in &self.l1_inverse) *a = b * c);
    }
}

impl L1 {
    pub fn new(mat: &CsrMatrix) -> Self {
        let diag_sqrt: Vec<f64> = mat.diag_iter().map(|a_ii| a_ii.unwrap().sqrt()).collect();
        let diag_sqrt_inv: Vec<f64> = diag_sqrt.iter().map(|val| val.recip()).collect();

        let l1_inverse: Vec<f64> = mat
            .outer_iterator()
            .enumerate()
            .map(|(i, row_vec)| {
                row_vec
                    .iter()
                    .map(|(j, val)| val.abs() * diag_sqrt[i] * diag_sqrt_inv[j])
                    .sum::<f64>()
                    .recip()
            })
            .collect();
        /* TODO should have L1 and L2 options
        let l1_inverse: Vec<f64> = mat
            .row_iter()
            .map(|row_vec| {
                let row_sum_abs: f64 = row_vec.values().iter().map(|val| val.abs()).sum();
                1.0 / row_sum_abs
            })
            .collect();
        */

        let l1_inverse: Vector = Vector::from(l1_inverse);
        //trace!("{:?}", l1_inverse.max());
        Self { l1_inverse }
    }
}

pub struct BlockL1 {
    partition: Arc<Partition>,
    blocks: Vec<Arc<dyn LinearOperator + Send + Sync>>,
    //blocks: Vec<Cholesky>,
    //blocks: Vec<CholeskyFactorized<OwnedRepr<f64>>>,
}

impl BlockL1 {
    // include tau?
    pub fn new(mat: &CsrMatrix, partition: Arc<Partition>) -> Self {
        let blocks: Vec<Arc<dyn LinearOperator + Send + Sync>> = partition
            .agg_to_node
            .par_iter()
            .map(|agg| {
                let block_size = agg.len();
                let agg: Vec<usize> = agg.iter().copied().collect();
                //let mut block = DMatrix::<f64>::zeros(block_size, block_size);
                let mut block = CooMatrix::new((block_size, block_size));

                for (ic, i) in agg.iter().copied().enumerate() {
                    assert!(mat.is_csr());
                    let mat_row_i = mat.outer_view(i).unwrap();
                    let a_ii = mat.get(i, i).unwrap();
                    for (j, val) in mat_row_i.iter() {
                        match agg.binary_search(&j) {
                            Ok(jc) => {
                                block.add_triplet(ic, jc, *val);
                            }
                            Err(_) => {
                                let a_jj = mat.get(j, j).unwrap();
                                //block[(ic, ic)] += (a_ii / a_jj).sqrt() * val.abs();
                                block.add_triplet(ic, ic, (a_ii / a_jj).sqrt() * val.abs());
                            }
                        }
                    }
                }
                let csr = block.to_csr();
                if csr.density() > 0.5 {
                    let dense = csr.to_dense();
                    let mut symmatrized = dense.clone() + dense.t();
                    symmatrized *= 0.5;

                    let chol = symmatrized.factorizec(UPLO::Upper).unwrap();

                    let f: Arc<dyn LinearOperator + Sync + Send> = Arc::new(move |in_vec: &Vector| -> Vector {
                        let mut out = in_vec.clone();
                        chol.solvec_inplace(&mut out).unwrap();
                        out
                    });
                    f
                } else {

                    let mut symmatrized = &csr.view() + &csr.transpose_view();
                    symmatrized.map_inplace(|v| v * 0.5);
                    let chol = Ldl::new()
                        .check_symmetry(sprs::SymmetryCheck::CheckSymmetry)
                        .check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry)
                        .fill_in_reduction(sprs::FillInReduction::CAMDSuiteSparse)
                        .numeric(symmatrized.view())
                        //.numeric(csr.view())
                        .expect("Constructing block Jacobi smoother failed because the restriction to an aggregate isn't SPD... Make sure A is SPD.");
                    let f: Arc<dyn LinearOperator + Sync + Send> = Arc::new(move |in_vec: &Vector| -> Vector {
                        chol.solve(in_vec)
                    });
                    f
                }
            })
        .collect();

        Self { partition, blocks }
    }
}

impl LinearOperator for BlockL1 {
    fn apply_mut(&self, r: &mut Vector) {
        // probably can do this without an alloc of a bunch of Vectors with unsafe
        let smoothed_parts: Vec<Vector> = self
            .partition
            .agg_to_node
            .par_iter()
            .enumerate()
            .map(|(i, agg)| {
                let r_part = Vector::from_iter(agg.iter().copied().map(|i| r[i]));

                //self.blocks[i].solvec_inplace(&mut r_part).unwrap();
                //self.blocks[i].solve(r_part)
                self.blocks[i].apply(&r_part)
            })
            .collect();

        // Could make this par iter but would need some hackery... probably not worth
        for (smoothed_part, agg) in smoothed_parts.iter().zip(self.partition.agg_to_node.iter()) {
            for (i, r_i) in agg.iter().copied().zip(smoothed_part.iter()) {
                r[i] = *r_i;
            }
        }
    }
}

// TODO abstract all to block smoother?? not sure what I would gain since there are only a couple
// block smoothers to consider for now.
pub struct BlockGaussSeidel {
    partition: Arc<Partition>,
    blocks: Vec<SymmetricGaussSeidel>,
    //forward: bool,
}

impl BlockGaussSeidel {
    // include tau?
    pub fn new(mat: &CsrMatrix, partition: Arc<Partition>) -> Self {
        let blocks: Vec<Arc<CsrMatrix>> = partition
            .agg_to_node
            .par_iter()
            .map(|agg| {
                let block_size = agg.len();
                let agg: Vec<usize> = agg.iter().copied().collect();
                let mut block = CooMatrix::new((block_size, block_size));

                for (ic, i) in agg.iter().copied().enumerate() {
                    let mat_row_i = mat.outer_view(i).unwrap();
                    let a_ii = mat.get(i, i).unwrap();
                    for (j, val) in mat_row_i.iter() {
                        match agg.binary_search(&j) {
                            Ok(jc) => {
                                block.add_triplet(ic, jc, *val);
                            }
                            Err(_) => {
                                let a_jj = mat.get(j, j).unwrap();
                                block.add_triplet(ic, ic, (a_ii / a_jj).sqrt() * val.abs());
                            }
                        }
                    }
                }
                let block = block.to_csr();
                Arc::new(block)
            })
            .collect();

        Self {
            partition,
            //blocks,
            // TODO forward then backward is better...
            blocks: blocks
                .into_iter()
                .map(|mat| SymmetricGaussSeidel::new(mat))
                .collect(),
            //forward: true,
        }
    }

    /*
    pub fn set_forward(&mut self, val: bool) {
    self.forward = val;
    }
    */
}

impl LinearOperator for BlockGaussSeidel {
    fn apply_mut(&self, r: &mut Vector) {
        // probably can do this without an alloc of a bunch of Vectors with unsafe
        let smoothed_parts: Vec<Vector> = self
            .partition
            .agg_to_node
            .par_iter()
            .enumerate()
            .map(|(i, agg)| {
                let mut r_part = Vector::from_iter(agg.iter().copied().map(|i| r[i]));
                self.blocks[i].apply_mut(&mut r_part);
                r_part
            })
            .collect();

        // Could make this par if reverse agg map was available... not sure if this is taking
        // meaningful time though
        for (smoothed_part, agg) in smoothed_parts.iter().zip(self.partition.agg_to_node.iter()) {
            for (i, r_i) in agg.iter().copied().zip(smoothed_part.iter()) {
                r[i] = *r_i;
            }
        }
    }
}

// TODO (idea) If solving for a block smoother iteratively, a permutation along with a diagonal
// shift is all the information we need to efficiently construct a solver on the fly for each
// block. This should significantly reduce the memory demands of the smoother. This should also be
// possible for the SGS version...
pub struct BlockL1Iterative {
    restriction: Arc<Partition>,
    blocks: Vec<Iterative>,
}

/*
   impl BlockL1Iterative {
// include tau?
pub fn new(mat: &CsrMatrix, restriction: Arc<CsrMatrix>) -> Self {
let blocks: Vec<CsrMatrix> = (0..restriction.rows())
.into_par_iter()
.map(|agg_idx| {
let r_row = restriction.row(agg_idx);
let agg = r_row.col_indices();
let block_size = agg.len();
let mut block = CooMatrix::new((block_size, block_size));

for (ic, i) in agg.iter().copied().enumerate() {
let mat_row_i = mat.row(i);
let a_ii = mat.get_entry(i, i).unwrap().into_value();
for (j, val) in mat_row_i
                        .col_indices()
                        .iter()
                        .copied()
                        .zip(mat_row_i.values().iter().copied())
                    {
                        match agg.binary_search(&j) {
                            Ok(jc) => {
                                block.push(ic, jc, val);
                            }
                            Err(_) => {
                                let a_jj = mat.get_entry(j, j).unwrap().into_value();
                                block.push(ic, ic, (a_ii / a_jj).sqrt() * val.abs());
                            }
                        }
                    }
                }
                CsrMatrix::<f64>::from(&block)
            })
            .collect();

        let blocks = blocks
            .into_iter()
            .map(|csr| {
                let block_size = csr.cols();
                let pc = L1::new(&csr);
                let ptr = Arc::new(csr);
                let epsilon = 1e-3;
                let solver = Iterative::new(ptr, Some(Vector::zeros(block_size)))
                    .with_tolerance(epsilon)
                    .with_max_iter(150)
                    .with_solver(IterativeMethod::ConjugateGradient)
                    //.with_log_interval(crate::solver::LogInterval::Iterations(51))
                    .with_preconditioner(Arc::new(pc));
                solver
            })
            .collect();

        Self {
            restriction,
            blocks,
        }
    }
}

impl LinearOperator for BlockL1Iterative {
    fn apply_mut(&self, r: &mut Vector) {
        // probably can do this without an alloc of a bunch of Vectors with unsafe
        let smoothed_parts: Vec<Vector> = (0..self.restriction.rows())
            .into_par_iter()
            .map(|i| {
                let row = self.restriction.row(i);
                let agg = row.col_indices();
                let mut r_part =
                    Vector::from_iterator(agg.len(), agg.iter().copied().map(|i| r[i]));
                self.blocks[i].apply_mut(&mut r_part);
                r_part
            })
            .collect();

        // Could make this par if reverse agg map was available... not sure if this is taking
        // meaningful time though
        for (smoothed_part, row) in smoothed_parts.iter().zip(self.restriction.row_iter()) {
            let agg = row.col_indices();
            for (i, r_i) in agg.iter().copied().zip(smoothed_part.iter()) {
                r[i] = *r_i;
            }
        }
    }
}
*/

pub struct ForwardGaussSeidel {
    mat: Arc<CsrMatrix>,
}

impl LinearOperator for ForwardGaussSeidel {
    fn apply_mut(&self, r: &mut Vector) {
        lsolve(self.mat.view(), r).unwrap();
    }
}

impl ForwardGaussSeidel {
    pub fn new(mat: Arc<CsrMatrix>) -> ForwardGaussSeidel {
        Self { mat }
    }
}

pub struct BackwardGaussSeidel {
    mat: Arc<CsrMatrix>,
}

impl LinearOperator for BackwardGaussSeidel {
    fn apply_mut(&self, r: &mut Vector) {
        usolve(self.mat.view(), r).unwrap();
    }
}

impl BackwardGaussSeidel {
    pub fn new(mat: Arc<CsrMatrix>) -> BackwardGaussSeidel {
        Self { mat }
    }
}

pub struct SymmetricGaussSeidel {
    diag: Vector,
    mat: Arc<CsrMatrix>,
}

impl LinearOperator for SymmetricGaussSeidel {
    fn apply_mut(&self, r: &mut Vector) {
        lsolve(self.mat.view(), r.view_mut()).unwrap();
        *r *= &self.diag;
        usolve(self.mat.view(), r.view_mut()).unwrap();
    }
}

impl SymmetricGaussSeidel {
    pub fn new(mat: Arc<CsrMatrix>) -> SymmetricGaussSeidel {
        let diag = mat.diag_iter().map(|v| *v.unwrap()).collect();
        Self { diag, mat }
    }
}

// TODO probably should do a multilevel gauss seidel and figure out to
// use the same code as L1
//      - test spd of precon again
pub struct Multilevel {
    pub hierarchy: Hierarchy,
    forward_smoothers: Vec<Arc<dyn LinearOperator + Sync + Send>>,
    backward_smoothers: Option<Vec<Arc<dyn LinearOperator + Sync + Send>>>,
    mu: usize,
}

impl LinearOperator for Multilevel {
    fn apply_mut(&self, r: &mut Vector) {
        self.init_w_cycle(r);
    }
}

impl Multilevel {
    // TODO builder pattern for Multilevel
    pub fn new(
        hierarchy: Hierarchy,
        solve_coarsest_exactly: bool,
        smoother: SmootherType,
        sweeps: usize,
        mu: usize,
    ) -> Self {
        let fine_mat = hierarchy.get_mat(0);
        let stationary = IterativeMethod::StationaryIteration;
        let pcg = IterativeMethod::ConjugateGradient;

        let build_smoother = |mat: Arc<CsrMatrix>,
                              smoother: SmootherType,
                              partition: Arc<Partition>,
                              transpose: bool|
         -> Arc<dyn LinearOperator + Send + Sync> {
            match smoother {
                SmootherType::L1 => Arc::new(L1::new(&mat)),
                // TODO no clone... use Arc
                //SmootherType::BlockL1 => Arc::new(BlockL1Iterative::new(&mat, restriction.clone())),
                SmootherType::BlockL1 => Arc::new(BlockL1::new(&mat, partition.clone())),
                SmootherType::GaussSeidel => {
                    if transpose {
                        Arc::new(BackwardGaussSeidel::new(mat))
                    } else {
                        Arc::new(ForwardGaussSeidel::new(mat))
                    }
                }
                SmootherType::BlockGaussSeidel => {
                    let block_gs = BlockGaussSeidel::new(&mat, partition.clone());
                    Arc::new(block_gs)
                    /*
                    if transpose {
                        let mut block_gs = BlockGaussSeidel::new(&mat, restriction.clone());
                        block_gs.set_forward(false);
                        Arc::new(block_gs)
                    } else {
                        let block_gs = BlockGaussSeidel::new(&mat, restriction.clone());
                        Arc::new(block_gs)
                    }
                    */
                }
            }
        };

        let r = metis_n(hierarchy.get_near_null(0), fine_mat.clone(), 64);
        //let r = hierarchy.get_partition(0).clone();

        let fine_smoother = build_smoother(fine_mat.clone(), smoother, Arc::new(r), false);
        let zeros = Vector::from(vec![0.0; fine_mat.cols()]);
        let forward_solver = Iterative::new(fine_mat.clone(), Some(zeros.clone()))
            .with_solver(stationary)
            .with_max_iter(sweeps)
            .with_preconditioner(fine_smoother)
            .with_tolerance(1e-12);
        let mut forward_smoothers: Vec<Arc<dyn LinearOperator + Send + Sync>> =
            vec![Arc::new(forward_solver)];

        let backward_smoothers = None;
        /*
        let mut backward_smoothers: Option<Vec<Arc<dyn LinearOperator + Send + Sync>>> =
            match smoother {
                SmootherType::L1 => None,
                SmootherType::BlockL1 => None,
                SmootherType::GaussSeidel => {
                    let backward_smoother = Arc::new(BackwardGaussSeidel::new(fine_mat.clone()));
                    let backward_solver = Iterative::new(fine_mat.clone(), Some(zeros))
                        .with_solver(stationary)
                        .with_max_iter(sweeps)
                        .with_preconditioner(backward_smoother)
                        .with_tolerance(1e-12);
                    Some(vec![Arc::new(backward_solver)])
                }
                SmootherType::BlockGaussSeidel => None,
            };
        */

        let coarse_index = hierarchy.get_coarse_mats().len() - 1;
        forward_smoothers.extend(
            hierarchy
                .get_coarse_mats()
                .iter()
                .enumerate()
                .map(|(i, mat)| {
                    let solver: Arc<dyn LinearOperator + Send + Sync>;
                    if i == coarse_index && solve_coarsest_exactly {
                        if mat.rows() < 1000 {
                            solver = Arc::new(Direct::new(&mat));
                        } else {
                            let pc = Arc::new(L1::new(mat));
                            solver = Arc::new(
                                Iterative::new(mat.clone(), Some(Vector::zeros(mat.cols())))
                                    .with_solver(pcg)
                                    .with_max_iter(10000)
                                    .with_preconditioner(pc)
                                    .with_tolerance(1e-8),
                            );
                        }
                    } else {
                        let r = metis_n(hierarchy.get_near_null(i + 1), mat.clone(), 64);
                        //let r = hierarchy.get_partition(i + 1).clone();
                        let smoother = build_smoother(mat.clone(), smoother, Arc::new(r), false);
                        let zeros = Vector::from(vec![0.0; mat.cols()]);
                        solver = Arc::new(
                            Iterative::new(mat.clone(), Some(zeros))
                                .with_solver(stationary)
                                .with_max_iter(sweeps)
                                .with_preconditioner(smoother)
                                .with_tolerance(0.0),
                        );
                    }
                    solver
                }),
        );

        /*
        if let Some(ref mut smoothers) = &mut backward_smoothers {
            smoothers.extend(
                hierarchy
                    .get_matrices()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i < coarse_index)
                    .map(|(i, mat)| {
                        let smoother = build_smoother(
                            mat.clone(),
                            smoother,
                            Arc::new(hierarchy.restriction_matrices[i + 1].clone()),
                            true,
                        );
                        let zeros = Vector::from(vec![0.0; mat.cols()]);

                        Arc::new(
                            Iterative::new(mat.clone(), Some(zeros))
                                .with_solver(stationary)
                                .with_max_iter(sweeps)
                                .with_preconditioner(smoother)
                                .with_tolerance(0.0),
                        )
                    }),
            );
        }
        */

        Self {
            hierarchy,
            forward_smoothers,
            backward_smoothers,
            mu,
        }
    }

    pub fn get_hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    fn init_w_cycle(&self, r: &mut Vector) {
        let mut v = Vector::from(vec![0.0; r.len()]);
        // TODO mu params in builder
        if self.hierarchy.levels() > 2 {
            self.w_cycle_recursive(&mut v, &*r, 0);
        } else {
            self.w_cycle_recursive(&mut v, &*r, 0);
        }
        r.clone_from(&v);
    }

    fn w_cycle_recursive(&self, v: &mut Vector, f: &Vector, level: usize) {
        let levels = self.hierarchy.levels() - 1;
        if level == levels {
            self.forward_smoothers[level].apply_input(f, v);
        } else {
            let pt = self.hierarchy.get_restriction(level);
            let p = self.hierarchy.get_interpolation(level);
            let a = &*self.hierarchy.get_mat(level);

            self.forward_smoothers[level].apply_input(f, v);

            let f_coarse = spmv(pt, &(f - &spmv(a, v)));
            //let f_coarse = pt * &(f - &(a * &*v));
            let mut v_coarse = Vector::from(vec![0.0; f_coarse.len()]);
            for _ in 0..self.mu {
                self.w_cycle_recursive(&mut v_coarse, &f_coarse, level + 1);
                /*
                if level + 1 == levels {
                    break;
                }
                */
            }

            let interpolated = spmv(p, &v_coarse);
            //let interpolated = p * &v_coarse;
            *v += &interpolated;
            if let Some(backward_smoothers) = &self.backward_smoothers {
                backward_smoothers[level].apply_input(f, v)
            } else {
                self.forward_smoothers[level].apply_input(f, v)
            };
        }
    }
}

// clone should be fine here since everything is Arc
#[derive(Clone)]
pub struct Composite {
    mat: Arc<CsrMatrix>,
    components: Vec<Arc<Multilevel>>,
}

impl LinearOperator for Composite {
    fn apply_mut(&self, vec: &mut Vector) {
        self.apply_multiplicative(vec);
    }
}

impl Composite {
    pub fn new(mat: Arc<CsrMatrix>) -> Self {
        Self {
            mat,
            components: Vec::new(),
        }
    }

    pub fn new_with_components(mat: Arc<CsrMatrix>, components: Vec<Arc<Multilevel>>) -> Self {
        Self { mat, components }
    }

    fn apply_multiplicative(&self, v: &mut Vector) {
        let mut x = Vector::from(vec![0.0; v.len()]);
        let mut r = v.clone();
        let num_steps = 1;
        let total_comps = self.components().len() * 2 - 1;
        let mut current_comp: usize = 0;
        for component in self.components.iter() {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                current_comp += 1;
                r = &*v - spmv(&self.mat, &x);
                if r.norm() < f64::EPSILON {
                    v.clone_from(&x);
                    warn!(
                        "Composite application early termination at {} of {} components because residual norm: {:.2e}",
                        current_comp,
                        total_comps,
                        r.norm()
                    );
                    return;
                }
                //r = &*v - (&*self.mat * &x);
            }
        }
        for component in self.components.iter().rev().skip(1) {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                current_comp += 1;
                r = &*v - spmv(&self.mat, &x);
                if r.norm() < f64::EPSILON {
                    v.clone_from(&x);
                    warn!(
                        "Composite application early termination at {} of {} components because residual norm: {:.2e}",
                        current_comp,
                        total_comps,
                        r.norm()
                    );
                    return;
                }
                //r = &*v - (&*self.mat * &x);
            }
        }
        v.clone_from(&x);
    }

    pub fn push(&mut self, component: Arc<Multilevel>) {
        self.components.push(component);
    }

    pub fn rm_oldest(&mut self) {
        self.components.remove(0);
    }

    pub fn components(&self) -> &Vec<Arc<Multilevel>> {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Vec<Arc<Multilevel>> {
        &mut self.components
    }

    pub fn get_mat(&self) -> Arc<CsrMatrix> {
        self.mat.clone()
    }
}
