//! Definition of the `Preconditioner` trait as well as implementors
//! of said trait.

use crate::parallel_ops::{interpolate, spmm_csr_dense};
use crate::partitioner::Hierarchy;
use crate::solver::{lsolve, pcg, usolve};
use nalgebra::base::DVector;
use nalgebra_sparse::CsrMatrix;
use std::path::Path;

// TODO make a composite preconditioner and clean up the adaptive module to use this.
// composite pc will have a Vec<Box<dyn Preconditioner>>

pub trait Preconditioner {
    fn apply(&mut self, r: &mut DVector<f64>);
}

pub struct L1 {
    l1_inverse: DVector<f64>,
}

impl Preconditioner for L1 {
    fn apply(&mut self, r: &mut DVector<f64>) {
        r.component_mul_assign(&self.l1_inverse);
    }
}

impl L1 {
    pub fn new(mat: &CsrMatrix<f64>) -> Self {
        let l1_inverse: Vec<f64> = mat
            .row_iter()
            .map(|row_vec| {
                let row_sum_abs: f64 = row_vec.values().iter().map(|val| val.abs()).sum();
                1.0 / row_sum_abs
            })
            .collect();
        let l1_inverse: DVector<f64> = DVector::from(l1_inverse);
        Self { l1_inverse }
    }
}

pub struct PcgL1 {
    mat: CsrMatrix<f64>,
    steps: usize,
    smoother: L1,
}

impl PcgL1 {
    pub fn new(mat: CsrMatrix<f64>, steps: usize) -> Self {
        let smoother = L1::new(&mat);
        Self {
            mat,
            steps,
            smoother,
        }
    }
}

impl Preconditioner for PcgL1 {
    fn apply(&mut self, r: &mut DVector<f64>) {
        let mut x = DVector::from(vec![0.0; r.nrows()]);
        let _ = pcg(
            &self.mat,
            r,
            &mut x,
            self.steps,
            1e-12,
            &mut self.smoother,
            None,
        );
        r.copy_from(&x);
    }
}

pub struct ForwardGaussSeidel {
    lower_triangle: CsrMatrix<f64>,
}

impl Preconditioner for ForwardGaussSeidel {
    fn apply(&mut self, r: &mut DVector<f64>) {
        lsolve(&self.lower_triangle, r);
    }
}

impl ForwardGaussSeidel {
    pub fn new(mat: &CsrMatrix<f64>) -> ForwardGaussSeidel {
        let lower_triangle = mat.lower_triangle();
        Self { lower_triangle }
    }
}

pub struct BackwardGaussSeidel {
    upper_triangle: CsrMatrix<f64>,
}

impl Preconditioner for BackwardGaussSeidel {
    fn apply(&mut self, r: &mut DVector<f64>) {
        usolve(&self.upper_triangle, r);
    }
}

impl BackwardGaussSeidel {
    pub fn new(mat: &CsrMatrix<f64>) -> BackwardGaussSeidel {
        let upper_triangle = mat.upper_triangle();
        Self { upper_triangle }
    }
}

pub struct SymmetricGaussSeidel {
    diag: DVector<f64>,
    upper_triangle: CsrMatrix<f64>,
    lower_triangle: CsrMatrix<f64>,
}

impl Preconditioner for SymmetricGaussSeidel {
    fn apply(&mut self, r: &mut DVector<f64>) {
        lsolve(&self.lower_triangle, r);
        r.component_mul_assign(&self.diag);
        usolve(&self.upper_triangle, r);
    }
}

impl SymmetricGaussSeidel {
    pub fn new(mat: &CsrMatrix<f64>) -> SymmetricGaussSeidel {
        let (_, _, diag) = mat.diagonal_as_csr().disassemble();
        let diag = DVector::from(diag);

        let lower_triangle = mat.lower_triangle();
        let upper_triangle = mat.upper_triangle();
        Self {
            diag,
            upper_triangle,
            lower_triangle,
        }
    }
}

// TODO probably should do a multilevel gauss seidel and figure out to
// use the same code as L1
//      - test spd of precon again
pub struct Multilevel<'a, T> {
    // TODO could probably greatly improve cache locality if workspaces were shared by
    // everyone. Would either require to be passed in or be globally available with unsafe.
    // Another advantage here would be that the apply method on the PC trait wouldn't
    // have to borrow self as mutable anymore (I think) which simplifies all the
    // borrowing shenanagins.
    x_ks: Vec<DVector<f64>>,
    b_ks: Vec<DVector<f64>>,
    r_ks: Vec<DVector<f64>>,
    hierarchy: Hierarchy<'a>,
    forward_smoothers: Vec<T>,
    //_backward_smoothers: Vec<T>,  // Eventually should probably support non-symmetric smoothers
}

impl<'a> Preconditioner for Multilevel<'a, PcgL1> {
    fn apply(&mut self, r: &mut DVector<f64>) {
        let levels = self.hierarchy.levels() - 1;
        let p_ks = self.hierarchy.get_partitions();
        let pt_ks = self.hierarchy.get_interpolations();

        // reset the workspaces
        for ((xk, bk), rk) in self
            .x_ks
            .iter_mut()
            .zip(self.b_ks.iter_mut())
            .zip(self.r_ks.iter_mut())
        {
            xk.fill(0.0);
            bk.fill(0.0);
            rk.fill(0.0);
        }

        self.b_ks[0].copy_from(&*r);

        for level in 0..levels {
            //for _ in 0..self.smoothing_steps {
            self.r_ks[level].copy_from(&self.b_ks[level]);
            crate::parallel_ops::spmm_csr_dense(
                1.0,
                &mut self.r_ks[level],
                -1.0,
                &self.hierarchy[level],
                &self.x_ks[level],
            );

            self.forward_smoothers[level].apply(&mut self.r_ks[level]);
            self.x_ks[level] += &self.r_ks[level];
            //}

            //let r_k = &self.b_ks[level] - &(&mat_ks[level] * &self.x_ks[level]);
            self.r_ks[level].copy_from(&self.b_ks[level]);
            spmm_csr_dense(
                1.0,
                &mut self.r_ks[level],
                -1.0,
                &self.hierarchy[level],
                &self.x_ks[level],
            );
            //let p_t = p_ks[level + 1].transpose();
            //self.b_ks[level + 1] = &p_t * &r_k;
            spmm_csr_dense(
                0.0,
                &mut self.b_ks[level + 1],
                1.0,
                &pt_ks[level],
                &self.r_ks[level],
            );
        }

        // Solve the coarsest problem almost exactly
        let (_converged, _ratio) = pcg(
            &self.hierarchy[levels],
            &self.b_ks[levels],
            &mut self.x_ks[levels],
            150,
            1.0e-6,
            &mut self.forward_smoothers[levels],
            None, //Some(5),
        );

        /*
        if !converged {
            warn!("PCG didn't converge with final ratio: {:3e}", ratio);
        }
        */

        /*
        info!(
            "norm of coarse correction: {}",
            self.x_ks[levels].dot(&self.x_ks[levels])
        );
        */

        for level in (0..levels).rev() {
            //let interpolated_x = self.hierarchy.get_partition(level + 1) * &self.x_ks[level + 1];
            //self.x_ks[level] += &interpolated_x;
            interpolate(&mut self.r_ks[level], &self.x_ks[level + 1], &p_ks[level]);
            self.x_ks[level] += &self.r_ks[level];

            //for _ in 0..self.smoothing_steps {
            //let mut r_k = &self.b_ks[level] - &(self.hierarchy.get_matrix(level) * &self.x_ks[level]);
            self.r_ks[level].copy_from(&self.b_ks[level]);
            spmm_csr_dense(
                1.0,
                &mut self.r_ks[level],
                -1.0,
                &self.hierarchy[level],
                &self.x_ks[level],
            );

            self.forward_smoothers[level].apply(&mut self.r_ks[level]);
            self.x_ks[level] += &self.r_ks[level];
            //}
        }
        r.copy_from(&self.x_ks[0]);
    }
}

/*
impl TwoLevel {
    pub fn new(hierarchy: Hierarchy) -> Self {

    }
}
*/

impl<'a> Multilevel<'a, PcgL1> {
    // TODO this constructor should borrow mat when partitioner/hierarchy changes happen
    pub fn new(hierarchy: Hierarchy<'a>) -> Self {
        trace!("building multilevel smoothers");
        let mut forward_smoothers = vec![PcgL1::new(hierarchy[0].clone(), 3)];
        forward_smoothers.extend(
            hierarchy
                .get_matrices()
                .iter()
                .map(|mat| PcgL1::new(mat.clone(), 5))
                .collect::<Vec<_>>(),
        );

        let mut x_ks: Vec<DVector<f64>> = vec![DVector::from(vec![0.0; hierarchy[0].nrows()])];
        x_ks.extend(
            hierarchy
                .get_matrices()
                .iter()
                .map(|p| DVector::from(vec![0.0; p.nrows()]))
                .collect::<Vec<_>>(),
        );
        let b_ks = x_ks.clone();
        let r_ks = x_ks.clone();

        Self {
            x_ks,
            b_ks,
            r_ks,
            hierarchy,
            forward_smoothers,
        }
    }
}

pub struct Composite<'a> {
    mat: &'a CsrMatrix<f64>,
    components: Vec<Box<dyn Preconditioner + 'a>>,
    x: DVector<f64>,
    y: DVector<f64>,
    r_work: DVector<f64>,
    application: CompositeType,
}

pub enum CompositeType {
    Additive,
    Sequential,
}

impl<'a> Preconditioner for Composite<'a> {
    fn apply(&mut self, r: &mut DVector<f64>) {
        self.x.fill(0.0);
        self.r_work.copy_from(r);

        match self.application {
            CompositeType::Sequential => self.apply_sequential(r),
            CompositeType::Additive => self.apply_additive(r),
        }
    }
}

impl<'a> Composite<'a> {
    pub fn new(
        mat: &'a CsrMatrix<f64>,
        components: Vec<Box<dyn Preconditioner>>,
        application: CompositeType,
    ) -> Self {
        let dim = mat.nrows();
        let x = DVector::from(vec![0.0; dim]);
        let y = x.clone();
        let r_work = x.clone();
        Self {
            mat,
            components,
            x,
            y,
            r_work,
            application,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, p: P) {}

    fn apply_sequential(&mut self, r: &mut DVector<f64>) {
        for component in self.components.iter_mut() {
            self.y.copy_from(&self.r_work);
            component.apply(&mut self.y);
            self.x += &self.y;
            //*r -= self.mat * &y;
            self.r_work.copy_from(r);
            spmm_csr_dense(1.0, &mut self.r_work, -1.0, &self.mat, &self.x);
        }
        for component in self.components.iter_mut().rev() {
            self.y.copy_from(&self.r_work);
            component.apply(&mut self.y);
            self.x += &self.y;
            //*r -= self.mat * &y;
            self.r_work.copy_from(r);
            spmm_csr_dense(1.0, &mut self.r_work, -1.0, &self.mat, &self.x);
        }
        r.copy_from(&self.x);
    }

    fn apply_additive(&mut self, r: &mut DVector<f64>) {
        let num_components = self.components.len() as f64;
        for component in self.components.iter_mut() {
            self.y.copy_from(&self.r_work);
            component.apply(&mut self.y);
            self.x += &self.y;
        }
        self.x /= num_components;
        r.copy_from(&self.x);
    }

    pub fn push(&mut self, component: Box<dyn Preconditioner + 'a>) {
        self.components.push(component);
    }

    pub fn rm_oldest(&mut self) {
        self.components.remove(1);
    }

    pub fn components(&'a self) -> &Vec<Box<dyn Preconditioner + 'a>> {
        &self.components
    }

    pub fn components_mut(&'a mut self) -> &mut Vec<Box<dyn Preconditioner + 'a>> {
        &mut self.components
    }
}
