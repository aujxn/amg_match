//! Definition of the `LinearOperator` trait as well as implementors
//! of said trait.
//!

use crate::hierarchy::Hierarchy;
use crate::parallel_ops::spmv;
use crate::partitioner::{
    block_strength, metis_n, modularity_matching_partition, reduce_block, Partition,
};
use crate::solver::{Direct, Iterative, IterativeMethod};
use crate::{CooMatrix, CsrMatrix, Matrix, Vector};
use core::f64;
use ndarray::{par_azip, Axis};
use ndarray_linalg::{hstack, FactorizeC, Norm, SolveC, SVD, UPLO};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use sprs::linalg::trisolve::{lsolve_csr_dense_rhs as lsolve, usolve_csr_dense_rhs as usolve};
use sprs::{is_symmetric, FillInReduction};
use sprs_ldl::Ldl;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::usize;

/// A Trait for objects which can map from R^n to R^m that are linear. In linear algebra
/// applications, this is most things (matrices, preconditioners, solvers, etc.).
///
/// <div class="warning">For linear operators which really on some iterative process
/// (such as Stationary Linear Iterations) the `apply_input` method will use the provided output
/// Vector as a starting guess for the process. Although this is common in other numerical
/// libraries, the Default impl of `apply_input` doesn't do this so this behaviour could be error
/// prone / confusing and I will likely decide to change this in the future to require an explicit
/// call to `Iterative::with_initial_guess` before applying.
/// </div>
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

    fn apply2(&self, mat: &Matrix) -> Matrix {
        let cols: Vec<_> = mat
            .axis_iter(Axis(1))
            .map(|col| self.apply(&col.to_owned()))
            .collect();
        hstack(&cols).unwrap()
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

/* API here is weird to design
pub enum GaussSeidelDirection {
    Forward,
    Backward,
    Symmetric
}
*/

/// Smoother options for constructing preconditioners. All smoothers should be A-convergent as
/// preconditioners for Stationary Linear Iterations or Conjugate Gradient as long as the matrix
/// `A` is SPD.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum SmootherType {
    /// An L1-scaled Jacobi smoother. Application is a diagonal matrix so this smoother has a low
    /// memory and application cost but is not suitable for challenging matrices.
    L1,
    /// A high quality smoother with a low memory footprint. The symmetric variant makes an
    /// additional copy of the matrix diagonal but otherwise only requires the storage of the `Arc`
    /// to the matrix. The main downside to this choice is that is is strictly sequential in its
    /// current implementation. In the future, we might support colored variations with weakly
    /// parallel capabilities but currently the best way to take advantage of GS smoothing in
    /// parallel is to use the "diagonally compensated block" variant.
    GaussSeidel,
    /// Current implementation has no fill-in and creates a new matrix with the same sparsity
    /// pattern as the matrix and a copy of the matrix diagonal. Like `GaussSeidel` this involves
    /// sparse triangular solves which are inherently serial. Use within "diagonally compensated
    /// block" to achieve parallelism.
    IncompleteCholesky,
    /// A more advanced smoother which uses rank-1 corrections to the matrix to delete entries off
    /// of a prescribed block diagonal. Diagonal blocks are specified by partitioning the matrix
    /// into the specified number of blocks. Using the number of processors as the number of blocks
    /// will provide decent CPU saturation but processors that finish early will be waiting for the
    /// others to finish. Using many more blocks than the number of processors will eventually
    /// degrade the preconditioner but can result in better saturation. For using a pre-specified
    /// partition, don't use the provided helpers which consume this enum and build the multilevel
    /// operator using the `MultilevelBuilder`.
    ///
    /// Any solver / smoother which is A-convergent should still be A-convergent when used
    /// in this block form. Each block component is applied in a `rayon` parallel iterator, so by
    /// using partitions with at least as many aggregates as processors, this allows the use of
    /// higher quality sequential smoothers locally within a thread in a sort-of algebraic domain
    /// decomposition fashion.
    DiagonalCompensatedBlock(BlockSmootherType, usize),
}

/// The `SmootherType::DiagonalCompenstatedBlock` can be solved or smoothed with different options
/// specified here. For more customization you will have to build your smoother yourself.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum BlockSmootherType {
    /// Performs one symmetric Gauss-Seidel sweep per block.
    GaussSeidel,
    /// Decomposes each block into LDL^T with no fill-in. Resulting decomposition has same sparsity
    /// pattern as the block matrix created by removing connections between aggregates in the
    /// partition.
    IncompleteCholesky,
    /// If the sparsity of a block is below 30% then solve with sparse Cholesky decomposition but
    /// otherwise just use dense Cholesky decompositions. Warning: large blocks can consume lots of
    /// memory even when the sparse decomposition is done.
    AutoCholesky(FillInReduction),
    /// Solve each block by cacheing a sparse Cholesky decomposition of each block using the
    /// provided reordering. Warning: If blocks are large and the reordering approach doesn't
    /// effectively reduce fill in this can consume lots of memory.
    SparseCholesky(FillInReduction),
    /// Solve each block by cacheing a dense Cholesky decomposition of each block. Warning: if the
    /// blocks are large this can consume all available memory very quickly.
    DenseCholesky,
    /// Solve each block to a relative accuracy provided (1e-6 is probably okay most the time but
    /// problem dependent) iteratively with conjugate gradient.
    ConjugateGradient(f64),
}

pub fn build_smoother(
    mat: Arc<CsrMatrix>,
    smoother: SmootherType,
    near_null: Arc<Vector>,
    _transpose: bool,
    vdim: usize,
) -> Arc<dyn LinearOperator + Send + Sync> {
    match smoother {
        SmootherType::L1 => Arc::new(L1::new(&mat)),
        SmootherType::GaussSeidel => {
            /*
            if transpose {
                Arc::new(BackwardGaussSeidel::new(mat))
            } else {
                Arc::new(ForwardGaussSeidel::new(mat))
            }
            */
            Arc::new(SymmetricGaussSeidel::new(mat))
        }
        SmootherType::IncompleteCholesky => Arc::new(IncompleteCholesky::new(mat)),
        SmootherType::DiagonalCompensatedBlock(block_smoother, n_parts) => {
            /*
            let strength = block_strength(&mat, vdim);
            let near_null = Vector::ones(strength.rows());
            let partition = metis_n(&near_null, Arc::new(strength), n_parts);
            */
            let (_, strength, near_null) = reduce_block(&mat, &near_null, vdim);
            let partition = metis_n(&near_null, strength, n_parts);

            let mut node_to_agg = Vec::new();
            let mut agg_to_node = Vec::new();
            for agg_id in partition.node_to_agg {
                for _ in 0..vdim {
                    node_to_agg.push(agg_id);
                }
            }
            for agg in partition.agg_to_node {
                let mut new_agg = BTreeSet::new();
                for node_id in agg {
                    let new_base = node_id * vdim;
                    for offset in 0..vdim {
                        new_agg.insert(new_base + offset);
                    }
                }
                agg_to_node.push(new_agg);
            }
            let partition = Partition {
                mat: mat.clone(),
                node_to_agg,
                agg_to_node,
            };
            /*
            let max_size = (mat.rows() / n_parts) + 3;
            let target_cf = mat.rows() as f64 / (n_parts as f64);
            let partition = modularity_matching_partition(
                mat.clone(),
                &*near_null,
                target_cf,
                Some(max_size),
                vdim,
            );
            */

            Arc::new(BlockSmoother::new(
                &*mat,
                Arc::new(partition),
                block_smoother,
                vdim,
            ))
        }
    }
}

pub struct Identity;
impl Identity {
    pub fn new() -> Self {
        Self
    }
}

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
    pub l1_inverse: Vector,
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

pub struct BlockSmoother {
    partition: Arc<Partition>,
    blocks: Vec<Arc<dyn LinearOperator + Send + Sync>>,
}

impl LinearOperator for BlockSmoother {
    fn apply_mut(&self, r: &mut Vector) {
        // TODO probably can do this without a allocating of a bunch of Vectors with unsafe
        let smoothed_parts: Vec<Vector> = self
            .partition
            .agg_to_node
            .par_iter()
            .enumerate()
            .map(|(i, agg)| {
                let r_part = Vector::from_iter(agg.iter().copied().map(|i| r[i]));
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

impl BlockSmoother {
    // include tau?
    pub fn new(
        mat: &CsrMatrix,
        partition: Arc<Partition>,
        smoother: BlockSmootherType,
        vdim: usize,
    ) -> Self {
        let blocks: Vec<Arc<dyn LinearOperator + Send + Sync>> = partition
            .agg_to_node
            .par_iter()
            .map(|agg| {
                let mut csr = if vdim == 1 {
                    Self::diagonally_compensate(agg, mat)
                } else {
                    Self::diagonally_compensate_vector(agg, mat, vdim)
                };

                #[cfg(debug_assertions)]
                {
                    let transpose = mat.transpose_view().to_csr();

                    for (i, (csr_row, transposed_row)) in mat
                        .outer_iterator()
                        .zip(transpose.outer_iterator())
                        .enumerate()
                    {
                        for ((j, v), (jt, vt)) in csr_row.iter().zip(transposed_row.iter()) {
                            assert_eq!(j, jt);
                            if vt != v {
                                let rel_err = (v - vt).abs() / v.abs().max(vt.abs());
                                let abs_err = (v - vt).abs();
                                assert!(rel_err.min(abs_err) < 1e-12, "Symmetry check failed. A_{},{} is {:.3e} but A_{},{} is {:.3e}. Relative error: {:.3e}, Absolute error: {:.3e}", 
                                    i, j, v, j, i, vt,
                                    rel_err, abs_err);
                            }
                        }
                    }
                }

                // TODO maybe not needed / helpful and this implementation is lazy and inefficient
                if !is_symmetric(&csr) {
                    csr = &csr.view() + &csr.transpose_view();
                    csr.map_inplace(|v| v * 0.5);
                }

                match smoother {
                    BlockSmootherType::GaussSeidel => {
                        let block_smoother: Arc<dyn LinearOperator + Send + Sync> = Arc::new(SymmetricGaussSeidel::new(Arc::new(csr)));
                        block_smoother
                    },
                    BlockSmootherType::IncompleteCholesky=> {
                        let block_smoother: Arc<dyn LinearOperator + Send + Sync> = Arc::new(IncompleteCholesky::new(Arc::new(csr)));
                        block_smoother
                    },
                    BlockSmootherType::AutoCholesky(fill_in_reduction) => {
                        if csr.density() > 0.3 {
                            let dense = csr.to_dense();

                            let chol = dense.factorizec(UPLO::Upper).unwrap();

                            let f: Arc<dyn LinearOperator + Sync + Send> = Arc::new(move |in_vec: &Vector| -> Vector {
                                let mut out = in_vec.clone();
                                chol.solvec_inplace(&mut out).unwrap();
                                out
                            });
                            f
                        } else {
                            let chol = Ldl::new()
                                .check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry)
                                .fill_in_reduction(fill_in_reduction)
                                .numeric(csr.view())
                                .expect("Constructing block Jacobi smoother failed because the restriction to an aggregate isn't SPD... Make sure A is SPD.");
                            let f: Arc<dyn LinearOperator + Sync + Send> = Arc::new(move |in_vec: &Vector| -> Vector {
                                chol.solve(in_vec)
                            });
                            f
                        }
                    },
                    BlockSmootherType::DenseCholesky=> {
                        Arc::new(Direct::new(&Arc::new(csr)))
                    },
                    BlockSmootherType::SparseCholesky(fill_in_reduction)=> {

                        let chol = Ldl::new()
                            .check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry)
                            .fill_in_reduction(fill_in_reduction)
                            .numeric(csr.view())
                            .expect("Constructing block Jacobi smoother failed because the restriction to an aggregate isn't SPD... Make sure A is SPD.");
                        let f: Arc<dyn LinearOperator + Sync + Send> = Arc::new(move |in_vec: &Vector| -> Vector {
                            chol.solve(in_vec)
                        });
                        f
                    },
                    BlockSmootherType::ConjugateGradient(tolerance)=> {
                        // TODO (idea) If solving for a block smoother iteratively, a permutation along with a diagonal
                        // shift is all the information we need to efficiently construct a solver on the fly for each
                        // block. This should significantly reduce the memory demands of the smoother. This should also be
                        // possible for the SGS version or other approaches which don't require decompositions.
                        let op = Arc::new(csr);
                        let cg: Arc<dyn LinearOperator + Sync + Send>  = Arc::new(Iterative::new(op.clone(), None)
                            .with_solver(IterativeMethod::ConjugateGradient)
                            .without_max_iter()
                            .with_relative_tolerance(tolerance)
                            .with_absolute_tolerance(f64::EPSILON)
                            .with_preconditioner(Arc::new(SymmetricGaussSeidel::new(op))));
                        cg
                    },
                }
            })
            .collect();

        Self { partition, blocks }
    }

    fn diagonally_compensate(agg: &BTreeSet<usize>, mat: &CsrMatrix) -> CsrMatrix {
        assert!(mat.is_csr());
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
                        block.add_triplet(ic, ic, 0.5 * (a_ii / a_jj).sqrt() * val.abs());
                    }
                }
            }
        }

        block.to_csr()
    }

    fn diagonally_compensate_vector(
        agg: &BTreeSet<usize>,
        mat: &CsrMatrix,
        vdim: usize,
    ) -> CsrMatrix {
        assert!(mat.is_csr());
        let block_size = agg.len();
        assert_eq!(block_size % vdim, 0);
        let agg: Vec<usize> = agg.iter().copied().collect();
        let mut block = CooMatrix::new((block_size, block_size));

        let mut to_compensate = HashSet::new();

        for (ic, i) in agg.iter().copied().enumerate() {
            let mat_row_i = mat.outer_view(i).unwrap();
            //let a_ii = mat.get(i, i).unwrap();
            for (j, val) in mat_row_i.iter() {
                match agg.binary_search(&j) {
                    Ok(jc) => {
                        block.add_triplet(ic, jc, *val);
                    }
                    Err(_) => {
                        //let a_jj = mat.get(j, j).unwrap();
                        //block.add_triplet(ic, ic, (a_ii / a_jj).sqrt() * val.abs());
                        let ic_start = ic - (ic % vdim);
                        let i_start = i - (i % vdim);
                        let j_start = j - (j % vdim);
                        to_compensate.insert((ic_start, (i_start, j_start)));
                    }
                }
            }
        }

        for (ic, (i, j)) in to_compensate {
            let mut block_a_ij = Matrix::zeros((vdim, vdim));
            for i_off in 0..vdim {
                for j_off in 0..vdim {
                    if let Some(val) = mat.get(i + i_off, j + j_off) {
                        block_a_ij[[i_off, j_off]] -= val;
                    }
                }
            }

            let (u, s, _vt) = block_a_ij.svd(true, false).unwrap();
            let u = u.unwrap();
            //let vt = vt.unwrap();
            let s = Matrix::from_diag(&s);
            let usut = u.dot(&s.dot(&u.t()));
            //let vsvt = vt.t().dot(&s.dot(&vt));

            //let jc = agg.binary_search(&j).unwrap();
            for i_off in 0..vdim {
                for j_off in 0..vdim {
                    block.add_triplet(ic + i_off, ic + j_off, 0.5 * usut[[i_off, j_off]]);
                    //block.add_triplet(jc + i_off, jc + j_off, vsvt[[i_off, j_off]]);
                }
            }
        }

        block.to_csr()
    }
}

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
    pub fn new(mat: Arc<CsrMatrix>) -> Self {
        let diag = mat.diag_iter().map(|v| *v.unwrap()).collect();
        Self { diag, mat }
    }
}

pub struct IncompleteCholesky {
    diag: Vector,
    mat: Arc<CsrMatrix>,
}

impl LinearOperator for IncompleteCholesky {
    fn apply_mut(&self, r: &mut Vector) {
        lsolve(self.mat.view(), r.view_mut()).unwrap();
        *r *= &self.diag;
        usolve(self.mat.view(), r.view_mut()).unwrap();
    }
}

impl IncompleteCholesky {
    // TODO improvements could be made to this smoother:
    // (1) use permutation to minimize discarded fill
    // (2) allow backfill based on criterion
    // (3) don't really need to copy the entire matrix
    //      (easy) just use the existing matrix sparsity pattern with one new values array
    //      (medium) only store the diagonal compensation
    //      (hard) easy/medium only work with no backfill, otherwise would need to store the
    //      diagonal compensation and the backfill info
    pub fn new(mat: Arc<CsrMatrix>) -> Self {
        let ndofs = mat.rows();
        let mut lu = (*mat).clone();
        assert!(lu.is_csr());

        let mut diag: Vec<f64> = lu.diag_iter().map(|v| *v.unwrap()).collect();

        for i in 0..ndofs {
            let a_ii_inv = diag[i].recip();
            assert!(a_ii_inv > 0.0 && a_ii_inv.is_finite());

            let row_i: Vec<(usize, f64)> = lu
                .outer_view(i)
                .unwrap()
                .iter()
                .skip_while(|(j, _)| *j <= i)
                .map(|(j, v)| (j, *v))
                .collect();

            lu.outer_view_mut(i).unwrap().iter_mut().for_each(|(j, v)| {
                if j > i {
                    *v *= a_ii_inv;
                } else if j < i {
                    *v *= diag[j].recip();
                }
            });

            for (si, vi) in row_i.iter().copied() {
                for (sj, vj) in row_i.iter().copied() {
                    let sij = vi * vj * a_ii_inv;
                    if si == sj {
                        let v = diag[si];
                        assert!(v - sij > 0.0);
                        diag[si] -= sij;
                    } else {
                        match lu.get_mut(si, sj) {
                            Some(v) => {
                                *v -= sij;
                            }
                            None => {
                                let x_ii = diag[si];
                                let x_jj = diag[sj];
                                let tau2 = (x_ii / x_jj).sqrt();
                                assert!(tau2.is_finite());
                                diag[si] += sij.abs() * tau2;
                                diag[sj] += sij.abs() / tau2;
                            }
                        }
                    }
                }
            }
        }
        lu.diag_iter_mut().for_each(|v| *v.unwrap() = 1.0);
        Self {
            mat: Arc::new(lu),
            diag: diag.iter().map(|v| v.recip()).collect(),
        }
    }
}

#[derive(Clone)]
pub struct MultilevelBuilder {
    // TODO
    _fine_mat: Arc<CsrMatrix>, //.....
}
impl MultilevelBuilder {
    // TODO
    pub fn new(fine_mat: Arc<CsrMatrix>) -> Self {
        Self {
            _fine_mat: fine_mat,
        }
    }
    // TODO
    pub fn build(self) -> Multilevel {
        unimplemented!()
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

        let fine_smoother = build_smoother(
            fine_mat.clone(),
            smoother,
            hierarchy.get_near_null(0).clone(),
            false,
            hierarchy.vdims[0],
        );
        let zeros = Vector::from(vec![0.0; fine_mat.cols()]);
        let forward_solver = Iterative::new(fine_mat.clone(), Some(zeros.clone()))
            .with_solver(stationary)
            .with_max_iter(sweeps)
            .with_preconditioner(fine_smoother)
            .with_relative_tolerance(1e-8)
            .with_absolute_tolerance(f64::EPSILON);
        let mut forward_smoothers: Vec<Arc<dyn LinearOperator + Send + Sync>> =
            vec![Arc::new(forward_solver)];

        let backward_smoothers = None;

        let coarse_index = hierarchy.get_coarse_mats().len() - 1;
        forward_smoothers.extend(
            hierarchy
                .get_coarse_mats()
                .iter()
                .enumerate()
                .map(|(i, mat)| {
                    let solver: Arc<dyn LinearOperator + Send + Sync>;
                    let zeros = Vector::zeros(mat.cols());

                    if i == coarse_index && solve_coarsest_exactly {
                        if mat.density() > 0.3 || mat.rows() < 1000 {
                            // If the coarse problem is small or mostly dense then use dense direct
                            // decomposition method
                            solver = Arc::new(Direct::new(&mat));
                        } else {
                            // Otherwise solve with PCG to decent accuracy
                            let pc = Arc::new(L1::new(mat));
                            /*
                            let pc = build_smoother(
                                mat.clone(),
                                SmootherType::DiagonalCompensatedBlock(
                                    BlockSmootherType::GaussSeidel,
                                    16,
                                ),
                                hierarchy.get_near_nulls().last().unwrap().clone(),
                                false,
                                *hierarchy.vdims.last().unwrap(),
                            );
                            */
                            solver = Arc::new(
                                Iterative::new(mat.clone(), Some(zeros))
                                    .with_solver(pcg)
                                    .with_max_iter(10000)
                                    .with_preconditioner(pc)
                                    .with_relative_tolerance(1e-12)
                                    .with_absolute_tolerance(f64::EPSILON),
                            );
                        }
                    } else {
                        let smoother = build_smoother(
                            mat.clone(),
                            smoother,
                            hierarchy.get_near_null(i + 1).clone(),
                            false,
                            hierarchy.vdims[i + 1],
                        );
                        solver = Arc::new(
                            Iterative::new(mat.clone(), Some(zeros))
                                .with_solver(stationary)
                                .with_max_iter(sweeps)
                                .with_preconditioner(smoother)
                                .with_relative_tolerance(0.0),
                        );
                    }
                    solver
                }),
        );

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
            let mut v_coarse = Vector::zeros(f_coarse.len());
            for _ in 0..self.mu {
                self.w_cycle_recursive(&mut v_coarse, &f_coarse, level + 1);
            }

            let interpolated = spmv(p, &v_coarse);
            *v += &interpolated;

            // Ideally the smoothers have a transpose method which returns a 'view' referencing the
            // original smoother. All available non-symmetric smoothers should be capable of
            // efficiently applying their transpose and I can't think of a situation where we would
            // actually require two different objects...
            if let Some(backward_smoothers) = &self.backward_smoothers {
                backward_smoothers[level].apply_input(f, v)
            } else {
                self.forward_smoothers[level].apply_input(f, v)
            };
        }
    }
}

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
        for component in self.components.iter().rev() {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                current_comp += 1;
                r = &*v - spmv(&self.mat, &x);

                #[cfg(debug_assertions)]
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
            }
        }
        for component in self.components.iter().skip(1) {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                current_comp += 1;
                r = &*v - spmv(&self.mat, &x);

                #[cfg(debug_assertions)]
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
            }
        }
        v.clone_from(&x);
    }

    pub fn op_complexity(&self) -> f64 {
        let mut op_complexity = 0.0;
        let last_idx = self.components.len() - 1;
        for (i, component) in self.components.iter().enumerate() {
            if i == last_idx {
                op_complexity += component.hierarchy.op_complexity()
            } else {
                op_complexity += 2.0 * component.hierarchy.op_complexity()
            }
        }
        op_complexity
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
