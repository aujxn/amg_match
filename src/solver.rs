//! Implementation of various sparse matrix solvers for the
//! system `Ax=b`.

//use crate::parallel_ops::spmm_csr_dense;
use crate::{
    parallel_ops::spmm,
    preconditioner::{Identity, LinearOperator},
    utils::{format_duration, random_vec},
};
use nalgebra::{base::DVector, Dyn, FullPivLU};
use nalgebra_sparse::{convert::serial::convert_csr_dense, csr::CsrMatrix};
use serde::{Deserialize, Serialize};
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SolveInfo {
    pub converged: bool,
    pub initial_residual_norm: f64,
    pub final_relative_residual_norm: f64,
    pub iterations: usize,
    pub time: Duration,
    pub relative_residual_norm_history: Vec<f64>,
}

pub enum Solver {
    Iterative(Iterative),
    Direct(Direct),
}

impl Solver {
    pub fn solve_with_guess(
        &self,
        rhs: &DVector<f64>,
        initial_guess: &DVector<f64>,
    ) -> DVector<f64> {
        match self {
            Self::Direct(solver) => solver.lu.solve(rhs).unwrap(),
            Self::Iterative(solver) => solver.solve_with_guess(rhs, initial_guess).0,
        }
    }
}

pub struct Iterative {
    mat: Rc<CsrMatrix<f64>>, // maybe eventually this becomes linear operator also
    solver: IterativeMethod,
    preconditioner: Rc<dyn LinearOperator>,
    max_iter: Option<usize>,
    max_duration: Option<Duration>,
    tolerance: f64,
    initial_guess: DVector<f64>,
    log_interval: Option<LogInterval>,
}

pub struct Direct {
    lu: FullPivLU<f64, Dyn, Dyn>,
}

impl Direct {
    pub fn new(mat: &Rc<CsrMatrix<f64>>) -> Self {
        let dense = convert_csr_dense(mat);
        let lu = FullPivLU::new(dense);
        Self { lu }
    }
}

impl LinearOperator for Direct {
    fn apply_mut(&self, vec: &mut DVector<f64>) {
        assert!(self.lu.solve_mut(vec));
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        self.lu.solve(vec).unwrap()
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        let solution = self.lu.solve(in_vec).unwrap();
        out_vec.copy_from(&solution);
    }
}

impl Iterative {
    pub fn new(mat: Rc<CsrMatrix<f64>>, initial_guess: Option<DVector<f64>>) -> Self {
        let initial_guess = initial_guess.unwrap_or(random_vec(mat.ncols()));
        Self {
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
        self.solve_with_guess(rhs, &self.initial_guess)
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
                    "solver didn't converge on coarsest level\n\tfinal ratio: {:.2e}\n\ttarget ratio: {:.2e}\n\titerations: {}\n\ttime: {}\n\tmatrix size: {}",
                    solve_info.final_relative_residual_norm,
                    self.tolerance,
                    solve_info.iterations,
                    format_duration(&solve_info.time),
                    self.mat.ncols()
                );
            } else {
                trace!(
                    "Solved in {} iterations and {} with {:.2e} relative residual",
                    solve_info.iterations,
                    format_duration(&solve_info.time),
                    solve_info.final_relative_residual_norm
                );
            }
        }
        (solution, solve_info)
    }
}

// This implementation could be way better to avoid allocations
impl LinearOperator for Iterative {
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
            // TODO this is real bad
            if i < j {
                rhs[i] -= val * rhs[j];
            } else if i == j {
                rhs[i] /= val;
            }
        }
    }
}

/// Lower triangular solve for sparse matrices and dense rhs
pub fn lsolve(mat: &CsrMatrix<f64>, rhs: &mut DVector<f64>) {
    for (i, row) in mat.row_iter().enumerate() {
        for (&j, val) in row.col_indices().iter().zip(row.values().iter()) {
            // TODO this is real bad
            if i > j {
                rhs[i] -= val * rhs[j];
            } else if i == j {
                rhs[i] /= val;
            }
        }
    }
}

impl Iterative {
    /// Stationary iterative method based on the preconditioner. Solves the
    /// system Ax = b for x where 'mat' is A and 'rhs' is b. Common preconditioners
    /// include L1 smoother, forward/backward/symmetric Gauss-Seidel, and
    /// multilevel methods.
    fn stationary(&self, rhs: &DVector<f64>, x: &mut DVector<f64>) -> SolveInfo {
        let mat = &*self.mat;
        let mut r = rhs - &(mat * &*x);
        let r0 = r.dot(&r);
        let convergence_criterion = r0 * self.tolerance * self.tolerance;
        let norm0 = r0.sqrt();

        if self.log_interval.is_some() {
            trace!("Initial Residual Norm : {norm0:.3e}");
        }

        let mut convergence_history = Vec::new();
        let mut iter: usize = 0;
        let mut last_log = Instant::now();
        let start_time = Instant::now();

        loop {
            self.preconditioner.apply_mut(&mut r);
            *x += &r;
            r = rhs - spmm(mat, &*x);
            iter += 1;

            let r_norm_squared = r.dot(&r);
            let ratio = (r_norm_squared / r0).sqrt();
            self.check_log_interval(
                iter,
                &mut last_log,
                &start_time,
                r_norm_squared.sqrt(),
                ratio,
            );
            convergence_history.push(ratio);

            if r_norm_squared < convergence_criterion {
                return SolveInfo {
                    converged: true,
                    initial_residual_norm: norm0,
                    final_relative_residual_norm: ratio,
                    iterations: iter,
                    time: Instant::now() - start_time,
                    relative_residual_norm_history: convergence_history,
                };
            }

            if self.check_max_conditions(iter, start_time) {
                return SolveInfo {
                    converged: false,
                    initial_residual_norm: norm0,
                    final_relative_residual_norm: ratio.sqrt(),
                    iterations: iter,
                    time: Instant::now() - start_time,
                    relative_residual_norm_history: convergence_history,
                };
            }
        }
    }

    /// Preconditioned conjugate gradient. Solves the system Ax = b for x where
    /// 'mat' is A and 'rhs' is b. The preconditioner is a function that takes
    /// a residual (vector) and returns the action of the inverse preconditioner
    /// on that residual.
    fn pcg(&self, rhs: &DVector<f64>, x: &mut DVector<f64>) -> SolveInfo {
        let mat = &*self.mat;
        let mut r = rhs - spmm(mat, &*x);

        let mut r_bar = r.clone();
        self.preconditioner.apply_mut(&mut r_bar);
        let mut d = r.dot(&r_bar);
        let d0 = d;
        let converged_criterion = d0 * self.tolerance * self.tolerance;
        let norm0 = d0.sqrt();
        if self.log_interval.is_some() {
            trace!("Initial (Br, r): {:.3e}", d0)
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
                    initial_residual_norm: norm0,
                    final_relative_residual_norm: relative_residual_norm,
                    iterations: iter,
                    time: Instant::now() - start_time,
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

            let ratio = (d / d0).sqrt();
            self.check_log_interval(iter, &mut last_log, &start_time, d.sqrt(), ratio);
            convergence_history.push(ratio);

            if d < converged_criterion {
                return SolveInfo {
                    converged: true,
                    initial_residual_norm: norm0,
                    final_relative_residual_norm: ratio,
                    iterations: iter,
                    time: Instant::now() - start_time,
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
            let solve_time = Instant::now() - start_time;
            if solve_time > max_time {
                return true;
            }
        }
        false
    }

    fn check_log_interval(
        &self,
        iter: usize,
        last_log: &mut Instant,
        start_time: &Instant,
        r_norm: f64,
        relative: f64,
    ) {
        if let Some(log_iter) = &self.log_interval {
            let now = Instant::now();
            let elapsed = now - *start_time;
            let log_fn = || {
                trace!(
                    "\n\ttime: {}\n\titer: {}\n\tresidual norm: {:.3e}\n\trelative norm: {:.3e}",
                    format_duration(&elapsed),
                    iter,
                    r_norm,
                    relative
                );
            };
            match log_iter {
                LogInterval::Iterations(log_iter) => {
                    if iter % log_iter == 0 {
                        log_fn();
                    }
                }
                LogInterval::Time(duration) => {
                    if now - *last_log > *duration {
                        *last_log = now;
                        log_fn();
                    }
                }
            }
        }
    }
}
