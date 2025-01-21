//! Definition of the `LinearOperator` trait as well as implementors
//! of said trait.

use std::sync::Arc;

use crate::parallel_ops::spmm;
use crate::partitioner::{metis_n, Hierarchy};
use crate::solver::{lsolve, usolve, Direct, Iterative, IterativeMethod};
use nalgebra::base::DVector;
//use nalgebra::{Cholesky, DMatrix, Dyn};
use nalgebra_sparse::factorization::CscCholesky;
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

pub trait LinearOperator {
    fn apply_mut(&self, vec: &mut DVector<f64>);
    fn apply(&self, vec: &DVector<f64>) -> DVector<f64>;
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>);
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum SmootherType {
    L1,
    GaussSeidel,
    BlockL1,
    BlockGaussSeidel,
}

pub struct Identity;
impl Identity {
    pub fn new() -> Self {
        Self
    }
}

impl LinearOperator for Identity {
    fn apply_mut(&self, _vec: &mut DVector<f64>) {}

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        vec.clone()
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
    }
}

pub struct L1 {
    l1_inverse: DVector<f64>,
}

impl LinearOperator for L1 {
    fn apply_mut(&self, vec: &mut DVector<f64>) {
        vec.component_mul_assign(&self.l1_inverse);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        vec.component_mul(&self.l1_inverse)
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(&in_vec.component_mul(&self.l1_inverse))
    }
}

impl L1 {
    pub fn new(mat: &CsrMatrix<f64>) -> Self {
        let diag_sqrt: Vec<f64> = mat
            .diagonal_as_csr()
            .values()
            .iter()
            .map(|a_ii| a_ii.sqrt())
            .collect();
        let diag_sqrt_inv: Vec<f64> = diag_sqrt.iter().map(|val| val.recip()).collect();

        let l1_inverse: Vec<f64> = mat
            .row_iter()
            .enumerate()
            .map(|(i, row_vec)| {
                row_vec
                    .col_indices()
                    .iter()
                    .zip(row_vec.values().iter())
                    .map(|(j, val)| val.abs() * diag_sqrt[i] * diag_sqrt_inv[*j])
                    .sum::<f64>()
                    .recip()
            })
            .collect();
        /*
        let l1_inverse: Vec<f64> = mat
            .row_iter()
            .map(|row_vec| {
                let row_sum_abs: f64 = row_vec.values().iter().map(|val| val.abs()).sum();
                1.0 / row_sum_abs
            })
            .collect();
        */

        let l1_inverse: DVector<f64> = DVector::from(l1_inverse);
        //trace!("{:?}", l1_inverse.max());
        Self { l1_inverse }
    }
}

pub struct BlockL1 {
    // probably should be Arc to avoid dup with Hierarchy, or even better should probably be a two way
    // map aggs -> indices and indices -> aggs.
    restriction: Arc<CsrMatrix<f64>>,
    //blocks: Vec<Cholesky<f64, Dyn>>,
    blocks: Vec<CscCholesky<f64>>,
}

impl BlockL1 {
    // include tau?
    pub fn new(mat: &CsrMatrix<f64>, restriction: Arc<CsrMatrix<f64>>) -> Self {
        //let blocks: Vec<Cholesky<f64, Dyn>> = (0..restriction.nrows()).into_par_iter().map(|agg_idx| {
        let blocks: Vec<CscCholesky<f64>> = (0..restriction.nrows()).into_par_iter().map(|agg_idx| {
            let r_row = restriction.row(agg_idx);
            let agg = r_row.col_indices();
            let block_size = agg.len();
            //let mut block = DMatrix::<f64>::zeros(block_size, block_size);
            let mut block = CooMatrix::<f64>::new(block_size, block_size);

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
                            /*
                            // only need lower triangular for cholesky
                            if ic >= jc {
                                block[(ic, jc)] +=  val; 
                            }
                            */
                            block.push(ic, jc, val);
                        }
                        Err(_) => {
                            let a_jj = mat.get_entry(j, j).unwrap().into_value();
                            //block[(ic, ic)] += (a_ii / a_jj).sqrt() * val.abs();
                            block.push(ic, ic, (a_ii / a_jj).sqrt() * val.abs());
                        }
                    }
                }
            }
            //let cholesky = Cholesky::new(block).expect("Constructing block Jacobi smoother failed because the restriction to an aggregate isn't SPD... Make sure A is SPD.");
            let csc = CscMatrix::from(&block);
            let cholesky = CscCholesky::factor(&csc).expect("Constructing block Jacobi smoother failed because the restriction to an aggregate isn't SPD... Make sure A is SPD.");

            let n_tri = ((csc.nnz() - block_size) / 2) + block_size;
            let sparsity_reduction = (cholesky.l().nnz() as f64) / (n_tri as f64);
            trace!("sparsity reduction: {:.2}", sparsity_reduction);

            cholesky
        }).collect();

        Self {
            restriction,
            blocks,
        }
    }
}

impl LinearOperator for BlockL1 {
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        self.apply_mut(out_vec);
    }

    fn apply_mut(&self, r: &mut DVector<f64>) {
        // probably can do this without an alloc of a bunch of DVectors with unsafe
        let smoothed_parts: Vec<DVector<f64>> = (0..self.restriction.nrows())
            .into_par_iter()
            .map(|i| {
                let row = self.restriction.row(i);
                let agg = row.col_indices();
                let mut r_part =
                    DVector::from_iterator(agg.len(), agg.iter().copied().map(|i| r[i]));
                self.blocks[i].solve_mut(&mut r_part);
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

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out_vec = vec.clone();
        self.apply_mut(&mut out_vec);
        out_vec
    }
}

// TODO abstract all to block smoother?? not sure what I would gain since there are only a couple
// block smoothers to consider for now.
pub struct BlockGaussSeidel {
    // should probably be a two way map: aggs -> indices and indices -> aggs.
    restriction: Arc<CsrMatrix<f64>>,
    //blocks: Vec<Arc<CsrMatrix<f64>>>,
    blocks: Vec<SymmetricGaussSeidel>,
    //forward: bool,
}

impl BlockGaussSeidel {
    // include tau?
    pub fn new(mat: &CsrMatrix<f64>, restriction: Arc<CsrMatrix<f64>>) -> Self {
        let blocks: Vec<Arc<CsrMatrix<f64>>> = (0..restriction.nrows())
            .into_par_iter()
            .map(|agg_idx| {
                let r_row = restriction.row(agg_idx);
                let agg = r_row.col_indices();
                let block_size = agg.len();
                let mut block = CooMatrix::<f64>::new(block_size, block_size);

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
                let block = CsrMatrix::from(&block);
                Arc::new(block)
            })
            .collect();

        Self {
            restriction,
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
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        self.apply_mut(out_vec);
    }

    fn apply_mut(&self, r: &mut DVector<f64>) {
        // probably can do this without an alloc of a bunch of DVectors with unsafe
        let smoothed_parts: Vec<DVector<f64>> = (0..self.restriction.nrows())
            .into_par_iter()
            .map(|i| {
                let row = self.restriction.row(i);
                let agg = row.col_indices();
                let mut r_part =
                    DVector::from_iterator(agg.len(), agg.iter().copied().map(|i| r[i]));
                self.blocks[i].apply_mut(&mut r_part);
                /*
                if self.forward {
                    let gs = ForwardGaussSeidel::new(self.blocks[i].clone());
                    gs.apply_mut(&mut r_part);
                } else {
                    let gs = BackwardGaussSeidel::new(self.blocks[i].clone());
                    gs.apply_mut(&mut r_part);
                }
                */
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

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out_vec = vec.clone();
        self.apply_mut(&mut out_vec);
        out_vec
    }
}
pub struct BlockL1Iterative {
    restriction: Arc<CsrMatrix<f64>>, // probably should be Arc to avoid dup with Hierarchy
    blocks: Vec<Iterative>,
}

impl BlockL1Iterative {
    // include tau?
    pub fn new(mat: &CsrMatrix<f64>, restriction: Arc<CsrMatrix<f64>>) -> Self {
        let blocks: Vec<CsrMatrix<f64>> = (0..restriction.nrows())
            .into_par_iter()
            .map(|agg_idx| {
                let r_row = restriction.row(agg_idx);
                let agg = r_row.col_indices();
                let block_size = agg.len();
                let mut block = CooMatrix::new(block_size, block_size);

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
                let block_size = csr.ncols();
                let pc = L1::new(&csr);
                let ptr = Arc::new(csr);
                let epsilon = 1e-3;
                let solver = Iterative::new(ptr, Some(DVector::zeros(block_size)))
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
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        self.apply_mut(out_vec);
    }

    fn apply_mut(&self, r: &mut DVector<f64>) {
        // probably can do this without an alloc of a bunch of DVectors with unsafe
        let smoothed_parts: Vec<DVector<f64>> = (0..self.restriction.nrows())
            .into_par_iter()
            .map(|i| {
                let row = self.restriction.row(i);
                let agg = row.col_indices();
                let mut r_part =
                    DVector::from_iterator(agg.len(), agg.iter().copied().map(|i| r[i]));
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

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out_vec = vec.clone();
        self.apply_mut(&mut out_vec);
        out_vec
    }
}

pub struct ForwardGaussSeidel {
    mat: Arc<CsrMatrix<f64>>,
}

impl LinearOperator for ForwardGaussSeidel {
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        lsolve(&self.mat, out_vec);
    }
    fn apply_mut(&self, r: &mut DVector<f64>) {
        lsolve(&self.mat, r);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out_vec = vec.clone();
        lsolve(&self.mat, &mut out_vec);
        out_vec
    }
}

impl ForwardGaussSeidel {
    pub fn new(mat: Arc<CsrMatrix<f64>>) -> ForwardGaussSeidel {
        Self { mat }
    }
}

pub struct BackwardGaussSeidel {
    mat: Arc<CsrMatrix<f64>>,
}

impl LinearOperator for BackwardGaussSeidel {
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        usolve(&self.mat, out_vec);
    }
    fn apply_mut(&self, r: &mut DVector<f64>) {
        usolve(&self.mat, r);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out_vec = vec.clone();
        usolve(&self.mat, &mut out_vec);
        out_vec
    }
}

impl BackwardGaussSeidel {
    pub fn new(mat: Arc<CsrMatrix<f64>>) -> BackwardGaussSeidel {
        Self { mat }
    }
}

pub struct SymmetricGaussSeidel {
    diag: DVector<f64>,
    mat: Arc<CsrMatrix<f64>>,
}

impl LinearOperator for SymmetricGaussSeidel {
    fn apply_mut(&self, r: &mut DVector<f64>) {
        lsolve(&self.mat, r);
        r.component_mul_assign(&self.diag);
        usolve(&self.mat, r);
    }
    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out = vec.clone();
        self.apply_mut(&mut out);
        out
    }
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        self.apply_mut(out_vec);
    }
}

impl SymmetricGaussSeidel {
    pub fn new(mat: Arc<CsrMatrix<f64>>) -> SymmetricGaussSeidel {
        let (_, _, diag) = mat.diagonal_as_csr().disassemble();
        let diag = DVector::from(diag);

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
}

impl LinearOperator for Multilevel {
    fn apply_mut(&self, r: &mut DVector<f64>) {
        self.init_w_cycle(r);
    }
    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut vec_mut = vec.clone();
        self.apply_mut(&mut vec_mut);
        vec_mut
    }
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        self.apply_mut(out_vec);
    }
}

impl Multilevel {
    // TODO builder pattern for Multilevel
    pub fn new(
        hierarchy: Hierarchy,
        solve_coarsest_exactly: bool,
        smoother: SmootherType,
        sweeps: usize,
    ) -> Self {
        trace!("constructing new ML component");
        let fine_mat = hierarchy.get_mat(0);
        let stationary = IterativeMethod::StationaryIteration;
        let pcg = IterativeMethod::ConjugateGradient;

        let build_smoother = |mat: Arc<CsrMatrix<f64>>,
                              smoother: SmootherType,
                              restriction: Arc<CsrMatrix<f64>>,
                              transpose: bool|
         -> Arc<dyn LinearOperator + Send + Sync> {
            match smoother {
                SmootherType::L1 => Arc::new(L1::new(&mat)),
                // TODO no clone... use Arc
                SmootherType::BlockL1 => Arc::new(BlockL1Iterative::new(&mat, restriction.clone())),
                SmootherType::GaussSeidel => {
                    if transpose {
                        Arc::new(BackwardGaussSeidel::new(mat))
                    } else {
                        Arc::new(ForwardGaussSeidel::new(mat))
                    }
                }
                SmootherType::BlockGaussSeidel => {
                    let block_gs = BlockGaussSeidel::new(&mat, restriction.clone());
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

        let r = metis_n(&hierarchy.near_nulls[0], &fine_mat, 16);

        let fine_smoother = build_smoother(fine_mat.clone(), smoother, Arc::new(r), false);
        let zeros = DVector::from(vec![0.0; fine_mat.ncols()]);
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

        let coarse_index = hierarchy.get_matrices().len() - 1;
        forward_smoothers.extend(hierarchy.get_matrices().iter().enumerate().map(|(i, mat)| {
            let solver: Arc<dyn LinearOperator + Send + Sync>;
            if i == coarse_index && solve_coarsest_exactly {
                if mat.nrows() < 1000 {
                    solver = Arc::new(Direct::new(&mat));
                } else {
                    let pc = Arc::new(L1::new(mat));
                    solver = Arc::new(
                        Iterative::new(mat.clone(), Some(DVector::zeros(mat.ncols())))
                            .with_solver(pcg)
                            .with_max_iter(10000)
                            .with_preconditioner(pc)
                            .with_tolerance(1e-8),
                    );
                }
            } else {
                let r = metis_n(&hierarchy.near_nulls[i + 1], &mat, 16);
                let smoother = build_smoother(mat.clone(), smoother, Arc::new(r), false);
                let zeros = DVector::from(vec![0.0; mat.ncols()]);
                solver = Arc::new(
                    Iterative::new(mat.clone(), Some(zeros))
                        .with_solver(stationary)
                        .with_max_iter(sweeps)
                        .with_preconditioner(smoother)
                        .with_tolerance(0.0),
                );
            }
            solver
        }));

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
                        let zeros = DVector::from(vec![0.0; mat.ncols()]);

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
        }
    }

    pub fn get_hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    fn init_w_cycle(&self, r: &mut DVector<f64>) {
        let mut v = DVector::from(vec![0.0; r.nrows()]);
        // TODO mu params in builder
        let mu = 3;
        if self.hierarchy.levels() > 2 {
            self.w_cycle_recursive(&mut v, &*r, 0, mu);
        } else {
            self.w_cycle_recursive(&mut v, &*r, 0, 1);
        }
        r.copy_from(&v);
    }

    fn w_cycle_recursive(&self, v: &mut DVector<f64>, f: &DVector<f64>, level: usize, mu: usize) {
        let levels = self.hierarchy.levels() - 1;
        if level == levels {
            self.forward_smoothers[level].apply_input(f, v);
        } else {
            let p = self.hierarchy.get_partition(level);
            let pt = self.hierarchy.get_interpolation(level);
            let a = &*self.hierarchy.get_mat(level);

            self.forward_smoothers[level].apply_input(f, v);

            let f_coarse = spmm(pt, &(f - &spmm(a, v)));
            //let f_coarse = pt * &(f - &(a * &*v));
            let mut v_coarse = DVector::from(vec![0.0; f_coarse.nrows()]);
            for _ in 0..mu {
                self.w_cycle_recursive(&mut v_coarse, &f_coarse, level + 1, mu);
                /*
                if level + 1 == levels {
                    break;
                }
                */
            }

            let interpolated = spmm(p, &v_coarse);
            //let interpolated = p * &v_coarse;
            *v += interpolated;
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
    mat: Arc<CsrMatrix<f64>>,
    components: Vec<Arc<Multilevel>>,
    pub application: Application,
}

#[derive(Clone, Copy, Debug)]
pub enum Application {
    Multiplicative,
    Random,
}

impl LinearOperator for Composite {
    fn apply_mut(&self, vec: &mut DVector<f64>) {
        match self.application {
            Application::Multiplicative => self.apply_multiplicative(vec),
            Application::Random => self.apply_rand(vec),
        }
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut vec = vec.clone();
        match self.application {
            Application::Multiplicative => self.apply_multiplicative(&mut vec),
            Application::Random => self.apply_rand(&mut vec),
        }
        vec
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        match self.application {
            Application::Multiplicative => self.apply_multiplicative(out_vec),
            Application::Random => self.apply_rand(out_vec),
        }
    }
}

impl Composite {
    pub fn new(mat: Arc<CsrMatrix<f64>>) -> Self {
        Self {
            mat,
            components: Vec::new(),
            application: Application::Multiplicative,
        }
    }

    pub fn new_with_components(mat: Arc<CsrMatrix<f64>>, components: Vec<Arc<Multilevel>>) -> Self {
        Self {
            mat,
            components,
            application: Application::Multiplicative,
        }
    }

    fn apply_multiplicative(&self, v: &mut DVector<f64>) {
        let mut x = DVector::from(vec![0.0; v.nrows()]);
        let mut r = v.clone();
        let num_steps = 1;
        for component in self.components.iter() {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                r = &*v - spmm(&self.mat, &x);
                //r = &*v - (&*self.mat * &x);
            }
        }
        for component in self.components.iter().rev().skip(1) {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                r = &*v - spmm(&self.mat, &x);
                //r = &*v - (&*self.mat * &x);
            }
        }
        v.copy_from(&x);
    }

    fn apply_rand(&self, v: &mut DVector<f64>) {
        self.components
            .choose(&mut rand::thread_rng())
            .unwrap()
            .apply_mut(v)
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
}
