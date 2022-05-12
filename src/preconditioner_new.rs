use crate::partitioner::Hierarchy;
use crate::solver::{lsolve, usolve};
use nalgebra::base::DVector;
use nalgebra_sparse::CsrMatrix;

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

pub struct Multilevel<T> {
    x_ks: Vec<DVector<f64>>,
    b_ks: Vec<DVector<f64>>,
    hierarchy: Hierarchy,
    forward_smoothers: Vec<T>,
    backward_smoothers: Vec<T>,
    smoothing_steps: usize,
    decomp: nalgebra_lapack::LU<f64, nalgebra::Dynamic, nalgebra::Dynamic>,
}

impl Preconditioner for Multilevel<L1> {
    fn apply(&mut self, r: &mut DVector<f64>) {
        let levels = self.hierarchy.get_partitions().len();
        let p_ks = self.hierarchy.get_partitions();
        let mat_ks = self.hierarchy.get_matrices();
        self.b_ks[0].copy_from(&*r);

        for level in 0..levels {
            for _ in 0..self.smoothing_steps {
                let mut r_k = &self.b_ks[level] - &(&mat_ks[level] * &self.x_ks[level]);
                self.forward_smoothers[level].apply(&mut r_k);
                self.x_ks[level] += &r_k;
            }
            let p_t = p_ks[level].transpose();
            let r_k = &self.b_ks[level] - &(&mat_ks[level] * &self.x_ks[level]);
            self.b_ks[level + 1] = &p_t * &r_k;
        }

        self.x_ks[levels] = self.decomp.solve(&self.b_ks[levels]).unwrap();

        for level in (0..levels).rev() {
            let interpolated_x = self.hierarchy.get_partition(level) * &self.x_ks[level + 1];
            self.x_ks[level] += &interpolated_x;
            for _ in 0..self.smoothing_steps {
                let mut r_k =
                    &self.b_ks[level] - &(self.hierarchy.get_matrix(level) * &self.x_ks[level]);
                self.forward_smoothers[level].apply(&mut r_k);
                self.x_ks[level] += &r_k;
            }
        }
        r.copy_from(&self.x_ks[0]);
        for (xk, bk) in self.x_ks.iter_mut().zip(self.b_ks.iter_mut()) {
            xk.fill(0.0);
            bk.fill(0.0);
        }
    }
}

impl Multilevel<L1> {
    pub fn new(hierarchy: Hierarchy) -> Self {
        let mat_coarse = nalgebra::DMatrix::from(hierarchy.get_matrices().last().unwrap());
        trace!("decomposing coarse problem");
        //let decomp = mat_coarse.lu();
        let decomp = nalgebra_lapack::LU::new(mat_coarse);
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

        Self {
            x_ks,
            b_ks,
            hierarchy,
            forward_smoothers,
            backward_smoothers,
            smoothing_steps,
            decomp,
        }
    }
}
