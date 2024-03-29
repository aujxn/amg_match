//! Implementation of various sparse matrix solvers for the
//! system `Ax=b`.

//use crate::parallel_ops::spmm_csr_dense;
use crate::{parallel_ops::spmm, preconditioner::Preconditioner};
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use std::time::Instant;

/*
enum LogInterval {
    Iterations(usize),
    Time(Duration),
}
*/

/*
pub trait Solver {
fn solve(
    mat: &CsrMatrix<f64>,
    rhs: &DVector<f64>,
    x: &mut DVector<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: &dyn Preconditioner,
    log_convergence: Option<usize>,
) -> bool {
    fn (&self, r: &mut DVector<f64>);
}
*/

pub trait LogConvergence {
    type ConvLogger;
    fn callback(
        &mut self,
        iter: usize,
        initial_squared_residual: f64,
        squared_residual: f64,
        iterate: &DVector<f64>,
        residual: &DVector<f64>,
    );
}

pub struct ConvergenceLogger {
    pub initial_residual: f64,
    pub residual_norm_hist: Vec<f64>,
}

impl ConvergenceLogger {
    pub fn new() -> Self {
        Self {
            initial_residual: 0.0,
            residual_norm_hist: Vec::new(),
        }
    }

    pub fn data(self) -> (f64, Vec<f64>) {
        (self.initial_residual, self.residual_norm_hist)
    }
}

impl LogConvergence for ConvergenceLogger {
    type ConvLogger = ConvergenceLogger;
    fn callback(
        &mut self,
        _iter: usize,
        initial_squared_residual: f64,
        squared_residual: f64,
        _iterate: &DVector<f64>,
        _residual: &DVector<f64>,
    ) {
        self.initial_residual = initial_squared_residual.powf(0.5);
        self.residual_norm_hist.push(squared_residual.powf(0.5));
    }
}

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
    x: &mut DVector<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: &dyn Preconditioner,
    log_convergence: Option<usize>,
) -> (bool, f64, usize) {
    let mut r = rhs - &(mat * &*x);
    //let mut r = DVector::from(vec![0.0; rhs.nrows()]);
    //r.copy_from(rhs);
    //spmm_csr_dense(1.0, &mut r, -1.0, mat, &*x);
    let r0_norm = r.dot(&r);

    if log_convergence.is_some() {
        trace!("r0 norm : {r0_norm:.3e}");
    }

    let epsilon_squared = epsilon * epsilon;

    preconditioner.apply(&mut r);
    *x += &r;
    let mut ratio = 1.0;

    for iter in 1..max_iter + 1 {
        //r = rhs - &(mat * &*x);
        r = rhs - spmm(mat, &*x);
        let r_norm = r.dot(&r);
        ratio = r_norm / r0_norm;

        if let Some(log_iter) = log_convergence {
            if iter % log_iter == 0 {
                let ratio = (r_norm / r0_norm).sqrt();
                trace!("squared norm iter {iter}: {r_norm:.3e} relative error: {ratio:.3e}");
            }
        }

        if r_norm < epsilon_squared * r0_norm {
            if log_convergence.is_some() {
                trace!("converged in {iter} iterations\n");
            }
            return (true, ratio, iter);
        }

        preconditioner.apply(&mut r);
        *x += &r;
    }

    (false, ratio.sqrt(), max_iter)
}

/// Preconditioned conjugate gradient. Solves the system Ax = b for x where
/// 'mat' is A and 'rhs' is b. The preconditioner is a function that takes
/// a residual (vector) and returns the action of the inverse preconditioner
/// on that residual.
pub fn pcg(
    mat: &CsrMatrix<f64>,
    rhs: &DVector<f64>,
    x: &mut DVector<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: &dyn Preconditioner,
    log_convergence: Option<usize>,
    mut convergence_logger: Option<&mut ConvergenceLogger>,
) -> (bool, f64, usize) {
    //let mut r = DVector::from(vec![0.0; rhs.nrows()]);
    //let mut g = r.clone();

    //let mut r = rhs - mat * &*x;
    let mut r = rhs - spmm(mat, &*x);
    let d0 = r.dot(&r);
    if log_convergence.is_some() {
        trace!("initial residual: {d0:.3e}")
    }

    let mut r_bar = r.clone();
    preconditioner.apply(&mut r_bar);
    let mut d = r.dot(&r_bar);
    if log_convergence.is_some() {
        trace!("initial d (r * r_bar): {d:.3e}");
    }
    let mut p = r_bar.clone();

    let mut last_log = Instant::now();

    for i in 1..max_iter + 1 {
        //let mut g = mat * &p;
        let mut g = spmm(mat, &p);
        let alpha = d / p.dot(&g);
        if alpha < 0.0 {
            error!("alpha is negative: {alpha}");
        }
        g *= alpha;
        //x += &(alpha * &p);
        x.iter_mut()
            .zip(p.iter())
            .for_each(|(x_i, p_i)| *x_i += *p_i * alpha);

        r -= &g;

        r_bar.copy_from(&r);
        preconditioner.apply(&mut r_bar);
        let d_old = d;
        d = r.dot(&r_bar);
        if d < 0.0 {
            error!("preconditioner is not spd: {d}");
        }
        let mut d_report = r.dot(&r);

        if let Some(log_iter) = log_convergence {
            let now = Instant::now();
            if (now - last_log).as_secs() > log_iter as u64 {
                //r = rhs - mat * &*x;
                r = rhs - spmm(mat, &*x);
                //r.copy_from(rhs);
                //spmm_csr_dense(1.0, &mut r, -1.0, mat, &*x);
                // manufacture a solution and measure true error norm
                // in A, this should be strictly monotone. Also test with
                // Ax=0 and initial random guess.
                //
                // Make tests for preconditioners and composite for spd
                //
                // Make two level version with coarsening factor 8.0 or 27.0
                //
                // Test anisotropy matrices
                d_report = r.dot(&r);
                let ratio = (d_report / d0).sqrt();
                trace!("squared norm iter {i}: {d_report:.3e} relative norm: {ratio:.3e}");
                last_log = now;
            }
        }

        if let Some(ref mut logger) = convergence_logger {
            logger.callback(i, d0, d_report, &*x, &r);
        }

        if d_report < epsilon * epsilon * d0 {
            if log_convergence.is_some() {
                trace!("converged in {i} iterations\n");
            }
            return (true, (d_report / d0).sqrt(), i);
        }

        let beta = d / d_old;
        p *= beta;
        p += &r_bar;
    }

    //let r_final = rhs - mat * &*x;
    let r_final = rhs - spmm(mat, &*x);
    let ratio = (r_final.dot(&r_final) / d0).sqrt();
    (false, ratio, max_iter)
}
