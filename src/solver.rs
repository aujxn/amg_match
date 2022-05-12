use crate::preconditioner_new::Preconditioner;
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;

/// Upper triangular solve for sparse matrices and dense rhs
pub fn usolve(mat: &CsrMatrix<f64>, rhs: &mut DVector<f64>) {
    for i in (0..mat.nrows()).rev() {
        let row = mat.row(i);
        for (&j, val) in row
            .col_indices()
            .iter()
            .rev()
            .zip(row.values().iter().rev())
        {
            if i != j {
                rhs[i] -= val * rhs[j];
            } else {
                rhs[i] /= val;
            }
        }
    }
}

/// Lower triangular solve for sparse matrices and dense rhs
pub fn lsolve(mat: &CsrMatrix<f64>, rhs: &mut DVector<f64>) {
    for (i, row) in mat.row_iter().enumerate() {
        for (&j, val) in row.col_indices().iter().zip(row.values().iter()) {
            if i != j {
                rhs[i] -= val * rhs[j];
            } else {
                rhs[i] /= val;
            }
        }
    }
}

/// Stationary iterative method based on the preconditioner. Solves the
/// system Ax = b for x where 'mat' is A and 'rhs' is b. Common preconditioners
/// include L1 smoother, forward/backward/symmetric Gauss-Seidel, and
/// multilevel methods.
pub fn stationary(
    mat: &CsrMatrix<f64>,
    rhs: &DVector<f64>,
    initial_iterate: &DVector<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: &mut dyn Preconditioner,
) -> (DVector<f64>, bool) {
    let mut x = initial_iterate.clone();
    let mut r = rhs - &(mat * &x);
    let r0_norm = r.dot(&r);
    let epsilon_squared = epsilon * epsilon;

    for iter in 0..max_iter {
        r = rhs - &(mat * &x);
        let r_norm = r.dot(&r);

        if iter % 50 == 0 {
            trace!("squared norm iter {iter}: {r_norm}");
        }

        if r_norm < epsilon_squared * r0_norm {
            info!("converged in {iter} iterations\n");
            return (x, true);
        }

        preconditioner.apply(&mut r);
        x += &r;
    }

    (x, false)
}
/// Preconditioned conjugate gradient. Solves the system Ax = b for x where
/// 'mat' is A and 'rhs' is b. The preconditioner is a function that takes
/// a residual (vector) and returns the action of the inverse preconditioner
/// on that residual.
pub fn pcg<F>(
    mat: &CsrMatrix<f64>,
    rhs: &DVector<f64>,
    initial_iterate: &DVector<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: &F,
) -> (DVector<f64>, bool)
where
    F: Fn(&mut DVector<f64>),
{
    let mut x = initial_iterate.clone();
    let mut r = rhs - mat * &x;
    let mut r_bar = r.clone();
    preconditioner(&mut r_bar);
    let d0 = r.dot(&r_bar);
    let mut d = d0;
    let mut p = r_bar.clone();

    for i in 0..max_iter {
        let mut g = mat * &p;
        let alpha = d / p.dot(&g);
        g *= alpha;
        x += &(alpha * &p);
        r -= &g;
        r_bar = r.clone();
        preconditioner(&mut r_bar);
        let d_old = d;
        d = r.dot(&r_bar);

        if i % 50 == 0 {
            trace!("squared norm iter {i}: {d}");
            r = rhs - mat * &x;
        }

        if d < epsilon * epsilon * d0 {
            info!("converged in {i} iterations\n");
            return (x, true);
        }

        let beta = d / d_old;
        p *= beta;
        p += &r_bar;
    }

    (x, false)
}
