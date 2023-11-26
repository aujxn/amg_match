//! Implementation of various sparse matrix solvers for the
//! system `Ax=b`.

//use crate::parallel_ops::spmm_csr_dense;
use crate::{
    parallel_ops::spmm,
    preconditioner::{Identity, LinearOperator},
    utils::random_vec,
};
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

#[derive(Copy, Clone, Debug)]
pub enum IterativeMethod {
    ConjugateGradient,
    StationaryIteration,
    //GMRES,
}

pub struct SolveInfo {
    pub converged: bool,
    pub initial_residual_norm: f64,
    pub final_relative_residual_norm: f64,
    pub iterations: usize,
    pub relative_residual_norm_history: Vec<f64>,
}

pub struct IterativeSolver {
    mat: Rc<CsrMatrix<f64>>, // maybe eventually this becomes linear operator also
    solver: IterativeMethod,
    preconditioner: Rc<dyn LinearOperator>,
    max_iter: Option<usize>,
    max_duration: Option<Duration>,
    tolerance: f64,
    initial_guess: DVector<f64>,
    log_interval: Option<LogInterval>,
}

impl IterativeSolver {
    pub fn new(mat: std::rc::Rc<CsrMatrix<f64>>, initial_guess: Option<DVector<f64>>) -> Self {
        let initial_guess = initial_guess.unwrap_or(random_vec(mat.ncols()));
        IterativeSolver {
            mat,
            solver: IterativeMethod::ConjugateGradient,
            preconditioner: Rc::new(Identity),
            max_iter: None,
            max_duration: None,
            tolerance: 1e-12,
            initial_guess,
            log_interval: None,
        }
    }

    pub fn with_solver(mut self, solver: IterativeMethod) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_preconditioner(mut self, preconditioner: Rc<dyn LinearOperator>) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    pub fn without_max_iter(mut self) -> Self {
        self.max_iter = None;
        self
    }

    pub fn with_max_duration(mut self, max_duration: Duration) -> Self {
        self.max_duration = Some(max_duration);
        self
    }

    pub fn without_max_duration(mut self) -> Self {
        self.max_duration = None;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_initial_guess(mut self, initial_guess: DVector<f64>) -> Self {
        self.initial_guess = initial_guess;
        self
    }

    pub fn with_log_interval(mut self, log_interval: LogInterval) -> Self {
        self.log_interval = Some(log_interval);
        self
    }

    pub fn without_log_interval(mut self) -> Self {
        self.log_interval = None;
        self
    }

    pub fn solve(&self, rhs: &DVector<f64>) -> (DVector<f64>, SolveInfo) {
        let mut solution = self.initial_guess.clone();

        let solve_info = match self.solver {
            IterativeMethod::ConjugateGradient => self.pcg(rhs, &mut solution),
            IterativeMethod::StationaryIteration => self.stationary(rhs, &mut solution),
        };

        if self.log_interval.is_some() {
            if !solve_info.converged {
                warn!(
                    "solver didn't converge on coarsest level\n\tfinal ratio: {:.2e}\n\ttarget ratio: {:.2e}\n\titerations: {}\n\tmatrix size: {}",
                    solve_info.final_relative_residual_norm,
                    self.tolerance,
                    solve_info.iterations,
                    self.mat.ncols()
                );
            } else {
                // TODO add solve time?
                trace!("Solved in {} iterations", solve_info.iterations);
            }
        }
        (solution, solve_info)
    }

    pub fn solve_with_guess(
        &self,
        rhs: &DVector<f64>,
        initial_guess: &DVector<f64>,
    ) -> (DVector<f64>, SolveInfo) {
        let mut solution = initial_guess.clone();

        let solve_info = match self.solver {
            IterativeMethod::ConjugateGradient => self.pcg(rhs, &mut solution),
            IterativeMethod::StationaryIteration => self.stationary(rhs, &mut solution),
        };

        if self.log_interval.is_some() {
            if !solve_info.converged {
                warn!(
                    "solver didn't converge on coarsest level\n\tfinal ratio: {:.2e}\n\ttarget ratio: {:.2e}\n\titerations: {}\n\tmatrix size: {}",
                    solve_info.final_relative_residual_norm,
                    self.tolerance,
                    solve_info.iterations,
                    self.mat.ncols()
                );
            } else {
                // TODO add solve time?
                trace!("Solved in {} iterations", solve_info.iterations);
            }
        }
        (solution, solve_info)
    }
}

// This implementation could be way better to avoid allocations
impl LinearOperator for IterativeSolver {
    fn apply_mut(&self, vec: &mut DVector<f64>) {
        let (solution, _solve_info) = self.solve(vec);
        vec.copy_from(&solution);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let (solution, _solve_info) = self.solve(vec);
        solution
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        let (solution, _solve_info) = self.solve(in_vec);
        out_vec.copy_from(&solution);
    }
}

// TODO move this to utils probably
pub enum LogInterval {
    Iterations(usize),
    Time(Duration),
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

impl IterativeSolver {
    /// Stationary iterative method based on the preconditioner. Solves the
    /// system Ax = b for x where 'mat' is A and 'rhs' is b. Common preconditioners
    /// include L1 smoother, forward/backward/symmetric Gauss-Seidel, and
    /// multilevel methods.
    fn stationary(&self, rhs: &DVector<f64>, x: &mut DVector<f64>) -> SolveInfo {
        let mat = &*self.mat;
        let epsilon = self.tolerance * self.tolerance;
        let mut r = rhs - &(mat * &*x);
        //let mut r = DVector::from(vec![0.0; rhs.nrows()]);
        //r.copy_from(rhs);
        //spmm_csr_dense(1.0, &mut r, -1.0, mat, &*x);
        let r0_norm = r.dot(&r);

        if self.log_interval.is_some() {
            trace!("r0 norm : {r0_norm:.3e}");
        }

        let mut convergence_history = Vec::new();

        self.preconditioner.apply_mut(&mut r);
        *x += &r;
        let mut ratio: f64 = 1.0;
        let mut iter: usize = 0;
        let mut last_log = Instant::now();
        let start_time = Instant::now();

        loop {
            if self.check_max_conditions(iter, start_time) {
                return SolveInfo {
                    converged: false,
                    initial_residual_norm: r0_norm.sqrt(),
                    final_relative_residual_norm: ratio.sqrt(),
                    iterations: iter,
                    relative_residual_norm_history: convergence_history,
                };
            }

            iter += 1;
            r = rhs - spmm(mat, &*x);
            let r_norm = r.dot(&r);
            ratio = r_norm / r0_norm;
            let relative_residual_norm = ratio.sqrt();
            self.check_log_interval(iter, &mut last_log, r0_norm.sqrt(), relative_residual_norm);
            convergence_history.push(relative_residual_norm);

            if r_norm < epsilon * r0_norm {
                return SolveInfo {
                    converged: true,
                    initial_residual_norm: r0_norm.sqrt(),
                    final_relative_residual_norm: relative_residual_norm,
                    iterations: iter,
                    relative_residual_norm_history: convergence_history,
                };
            }

            self.preconditioner.apply_mut(&mut r);
            *x += &r;
        }
    }

    /// Preconditioned conjugate gradient. Solves the system Ax = b for x where
    /// 'mat' is A and 'rhs' is b. The preconditioner is a function that takes
    /// a residual (vector) and returns the action of the inverse preconditioner
    /// on that residual.
    fn pcg(&self, rhs: &DVector<f64>, x: &mut DVector<f64>) -> SolveInfo {
        let mat = &*self.mat;
        let epsilon = self.tolerance * self.tolerance;
        let mut r = rhs - spmm(mat, &*x);
        let d0 = r.dot(&r);
        if self.log_interval.is_some() {
            trace!("initial residual: {d0:.3e}")
        }

        let mut r_bar = r.clone();
        self.preconditioner.apply_mut(&mut r_bar);
        let mut d = r.dot(&r_bar);
        if self.log_interval.is_some() {
            trace!("initial d (r * r_bar): {d:.3e}");
        }
        let mut p = r_bar.clone();

        let mut last_log = Instant::now();
        let mut iter = 0;
        let mut convergence_history = Vec::new();
        let start_time = Instant::now();

        loop {
            if self.check_max_conditions(iter, start_time) {
                let r_final = rhs - spmm(mat, &*x);
                let relative_residual_norm = (r_final.dot(&r_final) / d0).sqrt();
                return SolveInfo {
                    converged: false,
                    initial_residual_norm: d0.sqrt(),
                    final_relative_residual_norm: relative_residual_norm,
                    iterations: iter,
                    relative_residual_norm_history: convergence_history,
                };
            }

            iter += 1;

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
            self.preconditioner.apply_mut(&mut r_bar);
            let d_old = d;
            d = r.dot(&r_bar);
            if d < 0.0 {
                error!("preconditioner is not spd: {d}");
            }

            // TODO (testing)
            // manufacture a solution and measure true error norm
            // in A, this should be strictly monotone. Also test with
            // Ax=0 and initial random guess.
            //
            // Make tests for preconditioners and composite for spd

            let d_report = r.dot(&r);
            let ratio = (d_report / d0).sqrt();
            self.check_log_interval(iter, &mut last_log, d_report.sqrt(), ratio);
            convergence_history.push(ratio);

            if d_report < epsilon * d0 {
                return SolveInfo {
                    converged: true,
                    initial_residual_norm: d0.sqrt(),
                    final_relative_residual_norm: (d_report / d0).sqrt(),
                    iterations: iter,
                    relative_residual_norm_history: convergence_history,
                };
            }

            let beta = d / d_old;
            p *= beta;
            p += &r_bar;
        }
    }

    /// returns true if max condition is hit
    fn check_max_conditions(&self, iterations: usize, start_time: Instant) -> bool {
        if let Some(max_iter) = self.max_iter {
            if iterations >= max_iter {
                return true;
            }
        }
        if let Some(max_time) = self.max_duration {
            let solve_time = start_time - Instant::now();
            if solve_time > max_time {
                return true;
            }
        }
        false
    }

    fn check_log_interval(&self, iter: usize, last_log: &mut Instant, r_norm: f64, relative: f64) {
        if let Some(log_iter) = &self.log_interval {
            match log_iter {
                LogInterval::Iterations(log_iter) => {
                    if iter % log_iter == 0 {
                        trace!(
                            "iter {iter}\n\tresidual norm: {r_norm:.3e}\n\trelative norm: {relative:.3e}"
                        );
                    }
                }
                LogInterval::Time(duration) => {
                    let now = Instant::now();
                    if now - *last_log > *duration {
                        trace!(
                            "iter {iter}\n\tresidual norm: {r_norm:.3e}\n\trelative norm: {relative:.3e}"
                        );
                        *last_log = now;
                    }
                }
            }
        }
    }
}
