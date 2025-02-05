//! Implementation of various sparse matrix solvers for the
//! system `Ax=b`.
use crate::parallel_ops::spmv;
use crate::{
    preconditioner::{Identity, LinearOperator},
    utils::format_duration,
};
use crate::{CsrMatrix, Vector};
use ndarray::OwnedRepr;
use ndarray_linalg::{cholesky::*, Norm};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
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

// TODO this is so janky with handling iterative and direct with a new struct.
// The LinearOperator trait should probably have a way to set the initial guess...
// Even if this doesn't make sense for all linear operators.
impl Solver {
    pub fn solve_with_guess(&self, rhs: &Vector, initial_guess: &Vector) -> Vector {
        match self {
            Self::Direct(solver) => solver.cho_factor.solvec(rhs).unwrap(),
            Self::Iterative(solver) => solver.solve_with_guess(rhs, initial_guess).0,
        }
    }
}

pub struct Iterative {
    mat: Arc<CsrMatrix>, // maybe eventually this becomes linear operator also
    solver: IterativeMethod,
    preconditioner: Arc<dyn LinearOperator + Send + Sync>,
    max_iter: Option<usize>,
    max_duration: Option<Duration>,
    tolerance: f64,
    initial_guess: Vector,
    log_interval: Option<LogInterval>,
    //optimized: bool,
}

pub struct Direct {
    cho_factor: CholeskyFactorized<OwnedRepr<f64>>,
}

impl Direct {
    pub fn new(mat: &Arc<CsrMatrix>) -> Self {
        let cho_factor = mat.to_dense().factorizec(UPLO::Upper).unwrap();
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
            tolerance: 1e-12,
            initial_guess,
            log_interval: None,
            //optimized: false,
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

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
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

    /*
    pub fn optimized(mut self) -> Self {
        // TODO problably want to check for conflicting settings...
        self.optimized = true;
        self
    }

    pub fn featured(mut self) -> Self {
        self.optimized = false;
        self
    }
    */

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
                warn!(
                    "solver didn't converge on coarsest level\n\tfinal ratio: {:.2e}\n\ttarget ratio: {:.2e}\n\titerations: {}\n\ttime: {}\n\tmatrix size: {}",
                    solve_info.final_relative_residual_norm,
                    self.tolerance,
                    solve_info.iterations,
                    format_duration(&solve_info.time),
                    self.mat.cols()
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
        let mut r = rhs - &(mat * &*x);
        let test_norm = r.norm();
        let mut r0 = r.clone();
        self.preconditioner.apply_mut(&mut r0);
        //let r0 = r.dot(&r);
        let norm0 = r0.norm();
        //let r0 = r0.dot(&r0);
        //let convergence_criterion = r0 * self.tolerance * self.tolerance;

        if self.log_interval.is_some() {
            trace!("Initial Residual Norm / pc r norm: {test_norm:.3e} {norm0:.3e}");
        }

        let mut convergence_history = Vec::new();
        let mut iter: usize = 0;
        let mut last_log = Instant::now();
        let start_time = Instant::now();

        loop {
            self.preconditioner.apply_mut(&mut r);
            let r_norm = r.norm();
            let ratio = r_norm / norm0;
            *x += &r;
            r = rhs - spmv(mat, &*x);
            //r = rhs - (mat * &*x);
            iter += 1;

            //let ratio = (r_norm_squared / r0).sqrt();
            self.check_log_interval(iter, &mut last_log, &start_time, r_norm, ratio);
            if iter > 1 {
                convergence_history.push(ratio);
            }

            if ratio < self.tolerance {
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
                    final_relative_residual_norm: ratio,
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
            //let mut r = rhs - &(mat * &*x);
            let mut r = rhs - spmv(mat, &*x);
            // TODO remove this highly not optimal
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
        let mat = &*self.mat;
        let mut r = rhs - spmv(mat, &*x);
        //let mut r = rhs - (mat * &*x);

        let mut r_bar = r.clone();
        self.preconditioner.apply_mut(&mut r_bar);
        let mut d = r.dot(&r_bar);
        let d0 = d;
        if d0 < 1e-16 {
            return SolveInfo {
                converged: true,
                initial_residual_norm: d0,
                final_relative_residual_norm: 1.0,
                iterations: 0,
                time: Duration::ZERO,
                relative_residual_norm_history: Vec::new(),
            };
        }
        let converged_criterion = d0 * self.tolerance * self.tolerance;
        /*
        if converged_criterion < 1e-16 {
            warn!("Convergence criterion for PCG is very small... initial residual ((Br,r) = {:.3e}) might be nearly 0.0", d0);
        }
        */
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
                let r_final = rhs - spmv(mat, &*x);
                //let r_final = rhs - (mat * &*x);
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

            let mut g = spmv(mat, &p);
            //let mut g = mat * &p;
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

            r_bar.clone_from(&r);
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

    fn pcg_optimized_serial(&self, rhs: &Vector, x: &mut Vector) {
        let mat = &*self.mat;
        let mut r = rhs - (mat * &*x);

        let mut r_bar = r.clone();
        self.preconditioner.apply_mut(&mut r_bar);
        let mut d = r.dot(&r_bar);
        let d0 = d;
        if d0 < 1e-16 {
            return;
        }
        let converged_criterion = d0 * self.tolerance * self.tolerance;
        let mut p = r_bar.clone();

        loop {
            let mut g = mat * &p;
            let alpha = d / p.dot(&g);
            g *= alpha;
            //x += &(alpha * &p);
            x.iter_mut()
                .zip(p.iter())
                .for_each(|(x_i, p_i)| *x_i += *p_i * alpha);

            r -= &g;

            r_bar.clone_from(&r);
            self.preconditioner.apply_mut(&mut r_bar);
            let d_old = d;
            d = r.dot(&r_bar);

            if d < converged_criterion {
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
