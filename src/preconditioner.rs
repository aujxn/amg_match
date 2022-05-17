use crate::partitioner::Hierarchy;
use crate::solver::{lsolve, pcg_no_log, usolve};
use nalgebra::base::DVector;
use nalgebra_sparse::ops::{serial::spmm_csr_dense, Op};
use nalgebra_sparse::CsrMatrix;

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
    x_ks: Vec<DVector<f64>>,
    b_ks: Vec<DVector<f64>>,
    r_ks: Vec<DVector<f64>>,
    hierarchy: Hierarchy<'a>,
    forward_smoothers: Vec<T>,
    backward_smoothers: Vec<T>,
    smoothing_steps: usize,
}

impl<'a> Preconditioner for Multilevel<'a, L1> {
    fn apply(&mut self, r: &mut DVector<f64>) {
        let levels = self.hierarchy.len() - 1;
        let p_ks = self.hierarchy.get_partitions();
        let mat_ks = self.hierarchy.get_matrices();

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

        //self.b_ks[0] = p_ks[0].transpose() * &*r;
        spmm_csr_dense(
            0.0,
            &mut self.b_ks[0],
            1.0,
            Op::Transpose(&p_ks[0]),
            Op::NoOp(&*r),
        );

        for level in 0..levels {
            for _ in 0..self.smoothing_steps {
                self.r_ks[level].copy_from(&self.b_ks[level]);
                spmm_csr_dense(
                    1.0,
                    &mut self.r_ks[level],
                    -1.0,
                    Op::NoOp(&mat_ks[level]),
                    Op::NoOp(&self.x_ks[level]),
                );

                self.forward_smoothers[level].apply(&mut self.r_ks[level]);
                self.x_ks[level] += &self.r_ks[level];
            }
            //let r_k = &self.b_ks[level] - &(&mat_ks[level] * &self.x_ks[level]);
            self.r_ks[level].copy_from(&self.b_ks[level]);
            spmm_csr_dense(
                1.0,
                &mut self.r_ks[level],
                -1.0,
                Op::NoOp(&mat_ks[level]),
                Op::NoOp(&self.x_ks[level]),
            );
            //let p_t = p_ks[level + 1].transpose();
            //self.b_ks[level + 1] = &p_t * &r_k;
            spmm_csr_dense(
                0.0,
                &mut self.b_ks[level + 1],
                1.0,
                Op::Transpose(&p_ks[level + 1]),
                Op::NoOp(&self.r_ks[level]),
            );
        }

        let (coarse_solution, converged) = pcg_no_log(
            self.hierarchy.get_matrices().last().unwrap(),
            &self.b_ks[levels],
            &self.x_ks[levels],
            1000,
            1.0e-2,
            &mut self.forward_smoothers[levels],
        );
        if !converged {
            warn!("coarse solver didn't converge");
        }
        self.x_ks[levels].copy_from(&coarse_solution);

        for level in (0..levels).rev() {
            //let interpolated_x = self.hierarchy.get_partition(level + 1) * &self.x_ks[level + 1];
            //self.x_ks[level] += &interpolated_x;
            spmm_csr_dense(
                0.0,
                &mut self.r_ks[level],
                1.0,
                Op::NoOp(&p_ks[level + 1]),
                Op::NoOp(&self.x_ks[level + 1]),
            );
            self.x_ks[level] += &self.r_ks[level];

            for _ in 0..self.smoothing_steps {
                //let mut r_k = &self.b_ks[level] - &(self.hierarchy.get_matrix(level) * &self.x_ks[level]);
                self.r_ks[level].copy_from(&self.b_ks[level]);
                spmm_csr_dense(
                    1.0,
                    &mut self.r_ks[level],
                    -1.0,
                    Op::NoOp(&mat_ks[level]),
                    Op::NoOp(&self.x_ks[level]),
                );

                self.forward_smoothers[level].apply(&mut self.r_ks[level]);
                self.x_ks[level] += &self.r_ks[level];
            }
        }
        //*r = &p_ks[0] * &self.x_ks[0];
        spmm_csr_dense(0.0, r, 1.0, Op::NoOp(&p_ks[0]), Op::NoOp(&self.x_ks[0]));
    }
}

impl<'a> Multilevel<'a, L1> {
    // TODO this constructor should borrow mat when partitioner/hierarchy changes happen
    pub fn new(hierarchy: Hierarchy<'a>) -> Self {
        let smoothing_steps = 3;
        trace!("building multilevel smoothers");
        let forward_smoothers = hierarchy
            .get_matrices()
            .iter()
            .map(|mat| L1::new(mat))
            .collect::<Vec<_>>();
        let backward_smoothers = vec![];
        let x_ks: Vec<DVector<f64>> = hierarchy
            .get_matrices()
            .iter()
            .map(|p| DVector::zeros(p.nrows()))
            .collect();
        let b_ks = x_ks.clone();
        let r_ks = x_ks.clone();

        Self {
            x_ks,
            b_ks,
            r_ks,
            hierarchy,
            forward_smoothers,
            backward_smoothers,
            smoothing_steps,
        }
    }
}

pub struct Composite<'a> {
    mat: &'a CsrMatrix<f64>,
    components: Vec<Box<dyn Preconditioner + 'a>>,
    x: DVector<f64>,
    y: DVector<f64>,
    r_work: DVector<f64>,
}

impl<'a> Preconditioner for Composite<'a> {
    fn apply(&mut self, r: &mut DVector<f64>) {
        self.x.fill(0.0);
        self.r_work.copy_from(r);

        for component in self.components.iter_mut() {
            self.y.copy_from(&self.r_work);
            component.apply(&mut self.y);
            self.x += &self.y;
            //*r -= self.mat * &y;
            spmm_csr_dense(
                1.0,
                &mut self.r_work,
                -1.0,
                Op::NoOp(&self.mat),
                Op::NoOp(&self.y),
            );
        }
        for component in self.components.iter_mut().rev() {
            self.y.copy_from(&self.r_work);
            component.apply(&mut self.y);
            self.x += &self.y;
            //*r -= self.mat * &y;
            spmm_csr_dense(
                1.0,
                &mut self.r_work,
                -1.0,
                Op::NoOp(&self.mat),
                Op::NoOp(&self.y),
            );
        }
        r.copy_from(&self.x);
    }
}

impl<'a> Composite<'a> {
    pub fn new(mat: &'a CsrMatrix<f64>, components: Vec<Box<dyn Preconditioner>>) -> Self {
        let dim = mat.nrows();
        let x = DVector::zeros(dim);
        let y = DVector::zeros(dim);
        let r_work = DVector::zeros(dim);
        Self {
            mat,
            components,
            x,
            y,
            r_work,
        }
    }

    pub fn push(&mut self, component: Box<dyn Preconditioner + 'a>) {
        self.components.push(component);
    }

    pub fn components(&'a self) -> &Vec<Box<dyn Preconditioner + 'a>> {
        &self.components
    }

    pub fn components_mut(&'a mut self) -> &mut Vec<Box<dyn Preconditioner + 'a>> {
        &mut self.components
    }
}
