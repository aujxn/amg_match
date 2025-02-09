//! Implementation of various sparse matrix solvers for the
//! system `Ax=b`.
use crate::parallel_ops::spmv;
use crate::{
    preconditioner::{Identity, LinearOperator},
    utils::format_duration,
};
use crate::{CsrMatrix, Vector};
use ndarray::{Array2, OwnedRepr};
use ndarray_linalg::{cholesky::*, Norm};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use strum_macros::Display;

#[derive(Serialize, Deserialize, Copy, Clone, Debug, Display)]
#[non_exhaustive]
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
    pub final_absolute_residual_norm: f64,
    pub average_convergence_factor: f64,
    pub iterations: usize,
    pub time: Duration,
    pub relative_residual_norm_history: Vec<f64>,
}

pub enum Solver {
    Iterative(Iterative),
    Direct(Direct),
}

// TODO this API is so janky with handling iterative and direct with a new struct, needs re-work
impl Solver {
    pub fn solve_with_guess(&self, rhs: &Vector, initial_guess: &Vector) -> Vector {
        match self {
            Self::Direct(solver) => solver.cho_factor.solvec(rhs).unwrap(),
            Self::Iterative(solver) => solver.solve_with_guess(rhs, initial_guess).0,
        }
    }
}

/// A solver (or smoother) which solves the matrix system `Ax = b` using an iterative method.
/// Currently, the only 2 methods available are Stationary Linear Iteration (SLI) and (Preconditioned)
/// Conjugate Gradient (CG/PCG), both specified in `IterativeMethod` enum. Note that these methods are only
/// convergent under certain criteria. For CG `A` must be SPD and if preconditioned with `B` then
/// `B` must also be SPD. For SLI, by construction the error propagation operator given by `(I - BA)`
/// must be a contraction in the $A$ norm, i.e. there exists $\rho < 1.0$ such that for all $v \in
/// R^d$ we have $\rho \|(I - BA) (x - v)\|_A \leq \|x - v\|_A$.
///
/// <div class="warning">Even though the trait `LinearOperator` is implemented for this object that
/// doesn't actually mean the resulting operator is actually linear. Withholding the conversation
/// of numerical stabilities and floating point precision, the SLI variant is always linear provided
/// `B` is linear and symmetric provided that `A` and `B` are symmetric. PCG and CG are non-linear
/// solvers and are only analytically linear operators if run until the solution lies in the span
/// of the generated Krylov subspace. For practical purposes, if the tolerances are 'low enough'
/// and the solver can be used in cases where an SPD linear operator is expected (such as multigrid
/// coarse level smoothers and solvers).
/// </div>
pub struct Iterative {
    mat: Arc<CsrMatrix>, // maybe eventually this becomes linear operator also
    solver: IterativeMethod,
    preconditioner: Arc<dyn LinearOperator + Send + Sync>,
    max_iter: Option<usize>,
    max_duration: Option<Duration>,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    initial_guess: Vector,
    log_interval: Option<LogInterval>,
}

/// Solves the linear system `Ax = b` using a direct method. Currently the only option is by
/// (dense) Cholesky Decomposition, so don't use this on large matrices. This is primarily to be
/// used on the coarsest level solves of a multigrid hierarchy.
pub struct Direct {
    // TODO should probably have more options than just Cholesky
    cho_factor: CholeskyFactorized<OwnedRepr<f64>>,
}

impl Direct {
    pub fn new(mat: &Arc<CsrMatrix>) -> Self {
        let cho_factor = mat.to_dense().factorizec(UPLO::Upper).unwrap();
        Self { cho_factor }
    }

    pub fn from_dense(mat: &Array2<f64>) -> Self {
        let cho_factor = mat.factorizec(UPLO::Upper).unwrap();
        Self { cho_factor }
    }
}

impl LinearOperator for Direct {
    fn apply_mut(&self, vec: &mut Vector) {
        self.cho_factor.solvec_inplace(vec).unwrap();
    }

    fn apply(&self, vec: &Vector) -> Vector {
        self.cho_factor.solvec(vec).unwrap()
    }

    fn apply_input(&self, in_vec: &Vector, out_vec: &mut Vector) {
        out_vec.clone_from(in_vec);
        self.cho_factor.solvec_inplace(out_vec).unwrap();
    }
}

// TODO all this should probably move into the builder pattern
impl Iterative {
    pub fn new(mat: Arc<CsrMatrix>, initial_guess: Option<Vector>) -> Self {
        let initial_guess =
            initial_guess.unwrap_or(Vector::random(mat.cols(), Uniform::new(-1., 1.)));
        Self {
            mat,
            solver: IterativeMethod::ConjugateGradient,
            preconditioner: Arc::new(Identity),
            max_iter: None,
            max_duration: None,
            relative_tolerance: 1e-12,
            absolute_tolerance: 0.0,
            initial_guess,
            log_interval: None,
        }
    }

    pub fn set_op(mut self, mat: Arc<CsrMatrix>) -> Self {
        self.mat = mat;
        self
    }

    pub fn with_solver(mut self, solver: IterativeMethod) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_preconditioner(
        mut self,
        preconditioner: Arc<dyn LinearOperator + Send + Sync>,
    ) -> Self {
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

    pub fn with_relative_tolerance(mut self, tolerance: f64) -> Self {
        self.relative_tolerance = tolerance;
        self
    }

    pub fn with_absolute_tolerance(mut self, tolerance: f64) -> Self {
        self.absolute_tolerance = tolerance;
        self
    }

    pub fn with_initial_guess(mut self, initial_guess: Vector) -> Self {
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

    pub fn solve(&self, rhs: &Vector) -> (Vector, SolveInfo) {
        self.solve_with_guess(rhs, &self.initial_guess)
    }

    pub fn solve_optimized(&self, rhs: &Vector) -> Vector {
        let mut solution = self.initial_guess.clone();
        self.solve_optimized_with_guess(rhs, &mut solution);
        solution
    }

    pub fn solve_optimized_with_guess(&self, rhs: &Vector, solution: &mut Vector) {
        match self.solver {
            IterativeMethod::ConjugateGradient => self.pcg_optimized_serial(rhs, solution),
            IterativeMethod::StationaryIteration => {
                self.stationary_optimized_threaded(rhs, solution)
            }
        };
    }

    pub fn solve_with_guess(&self, rhs: &Vector, initial_guess: &Vector) -> (Vector, SolveInfo) {
        let mut solution = initial_guess.clone();

        let solve_info = match self.solver {
            IterativeMethod::ConjugateGradient => self.pcg(rhs, &mut solution),
            IterativeMethod::StationaryIteration => self.stationary(rhs, &mut solution),
        };

        if self.log_interval.is_some() {
            if !solve_info.converged {
                warn!("solver didn't converge!");
            }

            trace!(
                "{} results:\n\tSolve time: {} iterations in {}\n\tResidual norms: {:.2e} relative {:.2e} absolute\n\tAverage convergence: {:.2}",
                self.solver,
                solve_info.iterations,
                format_duration(&solve_info.time),
                solve_info.final_relative_residual_norm,
                solve_info.final_absolute_residual_norm,
                solve_info.average_convergence_factor
            );
        }
        (solution, solve_info)
    }
}

// This implementation could be way better to avoid allocations
impl LinearOperator for Iterative {
    fn apply_mut(&self, vec: &mut Vector) {
        let solution = self.solve_optimized(vec);
        vec.clone_from(&solution);
    }

    fn apply(&self, vec: &Vector) -> Vector {
        self.solve_optimized(vec)
    }

    fn apply_input(&self, in_vec: &Vector, out_vec: &mut Vector) {
        self.solve_optimized_with_guess(in_vec, out_vec);
    }
}

// TODO move this to utils probably
pub enum LogInterval {
    Iterations(usize),
    Time(Duration),
}

impl Iterative {
    /// Stationary iterative method based on the preconditioner. Solves the
    /// system Ax = b for x where 'mat' is A and 'rhs' is b. Common preconditioners
    /// include L1 smoother, forward/backward/symmetric Gauss-Seidel, and
    /// multilevel methods.
    fn stationary(&self, rhs: &Vector, x: &mut Vector) -> SolveInfo {
        let mat = &*self.mat;
        let mut r = rhs - &spmv(mat, &*x);
        self.preconditioner.apply_mut(&mut r);
        let norm0 = r.norm();
        let convergence_criterion =
            f64::max(norm0 * self.relative_tolerance, self.absolute_tolerance);

        if self.log_interval.is_some() {
            trace!("Initial Residual Norm ||B^-1 (b - Ax)||_2: {norm0:.3e}");
        }

        let mut convergence_history = Vec::new();
        let mut iter: usize = 0;
        let mut last_log = Instant::now();
        let start_time = Instant::now();

        loop {
            self.preconditioner.apply_mut(&mut r);
            let r_norm = r.norm();
            *x += &r;
            r = rhs - spmv(mat, &*x);
            iter += 1;

            let ratio = r_norm / norm0;
            self.check_log_interval(iter, &mut last_log, &start_time, r_norm, ratio);
            if iter > 1 {
                convergence_history.push(ratio);
            }

            if r_norm < convergence_criterion {
                return SolveInfo {
                    converged: true,
                    initial_residual_norm: norm0,
                    final_relative_residual_norm: ratio,
                    final_absolute_residual_norm: r_norm,
                    average_convergence_factor: ratio.powf(1.0 / iter as f64),
                    iterations: iter,
                    time: Instant::now() - start_time,
                    relative_residual_norm_history: convergence_history,
                };
            }

            if self.check_max_conditions(iter, start_time) {
                return SolveInfo {
                    converged: false,
                    initial_residual_norm: norm0,
                    final_relative_residual_norm: ratio,
                    final_absolute_residual_norm: r_norm,
                    average_convergence_factor: ratio.powf(1.0 / iter as f64),
                    iterations: iter,
                    time: Instant::now() - start_time,
                    relative_residual_norm_history: convergence_history,
                };
            }
        }
    }

    fn stationary_optimized_threaded(&self, rhs: &Vector, x: &mut Vector) {
        let mat = &*self.mat;
        let max_iter = self.max_iter.unwrap();

        // TODO handle case where x=0 (forward pass ML) so extra spmv is avoided
        for _ in 0..max_iter {
            let mut r = rhs - spmv(mat, &*x);
            #[cfg(debug_assertions)]
            if r.norm() < f64::EPSILON {
                warn!(
                    "Smoother application early termination because residual norm: {:.2e}",
                    r.norm()
                );
                return;
            }
            self.preconditioner.apply_mut(&mut r);
            *x += &r;
        }
    }

    /// Preconditioned conjugate gradient. Solves the system Ax = b for x where
    /// 'mat' is A and 'rhs' is b. The preconditioner is a function that takes
    /// a residual (vector) and returns the action of the inverse preconditioner
    /// on that residual.
    fn pcg(&self, rhs: &Vector, x: &mut Vector) -> SolveInfo {
        // TODO (testing)
        // manufacture a solution and measure true error norm
        // in A, this should be strictly monotone. Also test with
        // Ax=0 and initial random guess.
        //
        // Make tests for preconditioners and composite for spd

        let mat = &*self.mat;
        let mut r = rhs - spmv(mat, &*x);

        let mut r_bar = r.clone();
        self.preconditioner.apply_mut(&mut r_bar);
        let mut d = r.dot(&r_bar);
        let d0 = d;

        if d0 < 0.0 {
            error!(
                "The preconditioner is not positive definite. (Br, r) = {:.3e}",
                d0
            );

            return SolveInfo {
                converged: false,
                initial_residual_norm: d0,
                final_relative_residual_norm: 1.0,
                final_absolute_residual_norm: d0,
                average_convergence_factor: 1.0,
                iterations: 0,
                time: Duration::ZERO,
                relative_residual_norm_history: Vec::new(),
            };
        }

        let convergence_criterion = f64::max(
            d0 * self.relative_tolerance * self.relative_tolerance,
            self.absolute_tolerance * self.absolute_tolerance,
        );

        if d0 < convergence_criterion {
            return SolveInfo {
                converged: true,
                initial_residual_norm: d0.sqrt(),
                final_relative_residual_norm: 1.0,
                final_absolute_residual_norm: d0.sqrt(),
                average_convergence_factor: 1.0,
                iterations: 0,
                time: Duration::ZERO,
                relative_residual_norm_history: Vec::new(),
            };
        }

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
                let ratio = (d / d0).sqrt();
                return SolveInfo {
                    converged: false,
                    initial_residual_norm: d0.sqrt(),
                    final_relative_residual_norm: ratio,
                    final_absolute_residual_norm: d.sqrt(),
                    average_convergence_factor: ratio.powf(1.0 / iter as f64),
                    iterations: iter,
                    time: Instant::now() - start_time,
                    relative_residual_norm_history: convergence_history,
                };
            }

            iter += 1;

            let mut g = spmv(mat, &p);
            let alpha = d / p.dot(&g);
            if alpha < 0.0 {
                error!("alpha is negative: {alpha}");
            }
            g *= alpha;
            *x += &(alpha * &p);

            r -= &g;

            r_bar.clone_from(&r);
            self.preconditioner.apply_mut(&mut r_bar);
            let d_old = d;
            d = r.dot(&r_bar);
            if d < 0.0 {
                error!(
                    "The preconditioner is not positive definite. Iter {}: (Br, r) = {:.3e}",
                    iter, d0
                );

                return SolveInfo {
                    converged: false,
                    initial_residual_norm: d0,
                    final_relative_residual_norm: 1.0,
                    final_absolute_residual_norm: d0,
                    average_convergence_factor: 1.0,
                    iterations: 0,
                    time: Duration::ZERO,
                    relative_residual_norm_history: Vec::new(),
                };
            }

            let ratio = (d / d0).sqrt();
            self.check_log_interval(iter, &mut last_log, &start_time, d.sqrt(), ratio);
            convergence_history.push(ratio);

            if d < convergence_criterion {
                return SolveInfo {
                    converged: true,
                    initial_residual_norm: d0.sqrt(),
                    final_relative_residual_norm: ratio,
                    final_absolute_residual_norm: d.sqrt(),
                    average_convergence_factor: ratio.powf(1.0 / iter as f64),
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

    fn pcg_optimized_serial(&self, rhs: &Vector, x: &mut Vector) {
        let mat = &*self.mat;
        let mut r = rhs - (mat * &*x);

        let mut r_bar = r.clone();
        self.preconditioner.apply_mut(&mut r_bar);
        let mut d = r.dot(&r_bar);
        let d0 = d;
        let convergence_criterion = f64::max(
            d0 * self.relative_tolerance * self.relative_tolerance,
            self.absolute_tolerance * self.absolute_tolerance,
        );

        if d0 < convergence_criterion {
            return;
        }

        let mut p = r_bar.clone();

        loop {
            // Serial spmv with `Mul` trait, eventually move to threading profiles
            let mut g = mat * &p;
            let alpha = d / p.dot(&g);
            g *= alpha;
            *x += &(alpha * &p);
            r -= &g;

            r_bar.clone_from(&r);
            self.preconditioner.apply_mut(&mut r_bar);
            let d_old = d;
            d = r.dot(&r_bar);

            if d < convergence_criterion {
                return;
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

/*
pub fn lobpcg(
    a: Arc<dyn LinearOperator + Sync + Send>,
    b: Option<Arc<dyn LinearOperator + Sync + Send>>,
    pc: Option<Arc<dyn LinearOperator + Sync + Send>>,
    xs: &mut Array2<f64>,
    tol: f64,
    max_iter: usize,
) -> Vector {
    let pc = pc.unwrap_or(Arc::new(Identity));
    let b = b.unwrap_or(Arc::new(Identity));
    let nrows = xs.nrows();
    let m = xs.ncols();

    let mut ps = Array2::<f64>::zeros((nrows, m));
    let mut lambdas = Vector::from_elem(m, 1.0);
    let mut max_rnorm: f64 = 0.0;
    let mut sum_rnorms: f64 = 0.0;
    let mut ax = Array2::<f64>::zeros((nrows, m));
    let mut bx = Array2::<f64>::zeros((nrows, m));
    let mut ws = Array2::<f64>::zeros((nrows, m));

    //let log_interval = max_iter / 50;
    let log_interval = 1;

    for iter in 0..max_iter {
        let mut mgs = MGS::new(nrows, 1e-12);
        for col in xs.columns() {
            mgs.append(col);
            // if dependent then keep adding?? also eventually need to orthogonalize with B...
        }
        *xs = mgs.get_q();

        //ax.columns_mut()
        //.into_iter()
        ax.axis_iter_mut(Axis(0))
            .zip(xs.axis_iter(Axis(0)))
            .for_each(|(ax, x_vec)| {
                // ndarray is so frustrating sometimes but optimize later if really bad...
                // probably should make LinearOperator more abstract to apply to ArrayBase
                let out = a.apply(&x_vec.to_owned());
                ax.iter_mut().zip(out.iter()).for_each(|(a, b)| *a = *b);
                // why this doesn't work we will never know
                //par_azip!((a in ax, b in out) *a = b)
            });

        bx.columns_mut()
            .into_iter()
            .zip(xs.columns().into_iter())
            .for_each(|(ax, x_vec)| {
                let out = a.apply(&x_vec.to_owned());
                ax.iter_mut().zip(out.iter()).for_each(|(a, b)| *a = *b);
            });

        lambdas = ax
            .axis_iter(Axis(0))
            .zip(bx.axis_iter(Axis(0)))
            .zip(xs.axis_iter(Axis(0)))
            .map(|((ax_vec, bx_vec), x_vec)| {
                let denom = x_vec.inner(&bx_vec);
                assert!((1.0 - denom).abs() < 1e-12);
                x_vec.inner(&ax_vec) / denom
            })
            .collect();

        let rs = ax - (lambdas * xs.t()).t();
        /*
            .axis_iter(Axis(0))
            .zip(xs.axis_iter(Axis(0)))
            .zip(lambdas.iter().copied())
            .map(|((ax_vec, x_vec), lambda)| ax_vec - (&lambda * x_vec))
            .collect();
        */

        let r_norms = rs.map_axis(Axis(0), |r_vec| r_vec.norm());

        max_rnorm = r_norms
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        sum_rnorms = r_norms.iter().sum();
        if iter % log_interval == 0 {
            trace!(
                "Iter: {}, (sum, max) residual norm: ({:.2e}, {:.2e}), current max lambda: {:.2e}",
                iter,
                sum_rnorms,
                max_rnorm,
                lambdas[0]
            );
        }

        if max_rnorm < tol {
            trace!(
                "LOBPCG converged in {} iters with final max residual norm {:.2e} and max lambda {:.2e}",
                iter,
                max_rnorm,
                lambdas[0]
            );
            return lambdas;
        }

        //let ws = rs.map(|r_vec| pc.apply(&r_vec)).collect();

        ws.columns_mut()
            .into_iter()
            .zip(rs.columns().into_iter())
            .for_each(|(w_vec, r_vec)| {
                let out = pc.apply(&r_vec.to_owned());
                w_vec.iter_mut().zip(out.iter()).for_each(|(a, b)| *a = *b);
            });

        for col in rs.columns() {
            mgs.append(col);
        }
        for col in ws.columns() {
            mgs.append(col);
        }
        for col in ps.columns() {
            mgs.append(col);
        }

        let rayleigh_space = mgs.get_q();

        let cols: Vec<Vector> = rayleigh_space
            .column_iter()
            .map(|col| spmv(mat, &Vector::from(col)))
            .collect();

        let temp = DMatrix::from_columns(&cols);

        let mut projection = DMatrix::zeros(4 * m, 4 * m);
        projection.gemm_tr(1.0, &rayleigh_space, &temp, 1.0);

        let decomp = projection.symmetric_eigen();
        let eigs = decomp.eigenvalues;
        let eigvecs = decomp.eigenvectors;

        let mut indices = (0..(4 * m)).collect::<Vec<_>>();
        indices.sort_by(|idx_a, idx_b| eigs[*idx_b].partial_cmp(&eigs[*idx_a]).unwrap());

        /* This is correct impl according to paper but for some reason just setting the `ps` to the
        * previous `xs` works better
        ps.iter_mut()
            .zip(indices.iter().copied())
            .for_each(|(p_vec, idx)| {
                p_vec.copy_from(&Vector::from_element(nrows, 0.0));
                p_vec.gemv(
                    1.0,
                    &rayleigh_space.columns(m, 3 * m),
                    &eigvecs.column_part(idx, 3 * m),
                    1.0,
                );
            });
        */

        ps = *xs;
        xs.iter_mut()
            .zip(indices.into_iter())
            .for_each(|(x_vec, idx)| {
                x_vec.copy_from(&Vector::from_element(nrows, 0.0));
                x_vec.gemv(1.0, &rayleigh_space, &eigvecs.column(idx), 1.0);
            });
    }

    warn!(
        "LOBPCG didn't conver in {} iterations. Final (sum, max) residual norm: ({:.2e}, {:.2e}) and max lambda {:.2e}",
                max_iter,
                sum_rnorms,
                max_rnorm,
                lambdas[0]
            );

    lambdas
}
*/
