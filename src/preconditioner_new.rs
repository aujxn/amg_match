use crate::solver::{lsolve, usolve};
use nalgebra::base::DVector;
use nalgebra_sparse::CsrMatrix;

pub trait Preconditioner {
    fn apply(&self, r: &mut DVector<f64>);
}

pub struct L1 {
    l1_inverse: DVector<f64>,
}

impl Preconditioner for L1 {
    fn apply(&self, r: &mut DVector<f64>) {
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
    fn apply(&self, r: &mut DVector<f64>) {
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
    fn apply(&self, r: &mut DVector<f64>) {
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
    fn apply(&self, r: &mut DVector<f64>) {
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
