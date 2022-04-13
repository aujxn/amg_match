use ndarray::Array1;
use sprs::CsMat;

//TODO sgs, more smoothing, check symmetry with more smoothing
//then -> adaptivity
//TESTS: P*1 = 1, symmetry of precon, sum of nnz of all levels / nnz fine < 2 (~1 is best)

/// Stationary iterative method based on the preconditioner. Solves the
/// system Ax = b for x where 'mat' is A and 'rhs' is b. Common preconditioners
/// include L1 smoother, forward/backward/symmetric Gauss-Seidel, and
/// multilevel methods.
pub fn stationary<F>(
    mat: &CsMat<f64>,
    rhs: &Array1<f64>,
    initial_iterate: &Array1<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: &F,
) -> (Array1<f64>, bool)
where
    F: Fn(&mut Array1<f64>),
{
    let mut x = initial_iterate.clone();
    let mut r = rhs - &(mat * &x);
    let r0_norm = r.t().dot(&r);
    let epsilon_squared = epsilon * epsilon;

    for iter in 0..max_iter {
        r = rhs - &(mat * &x);
        let r_norm = r.t().dot(&r);

        if iter % 50 == 0 {
            trace!("squared norm iter {iter}: {r_norm}");
        }

        if r_norm < epsilon_squared * r0_norm {
            info!("converged in {iter} iterations\n");
            return (x, true);
        }

        preconditioner(&mut r);
        x += &r;
    }

    (x, false)
}
/// Preconditioned conjugate gradient. Solves the system Ax = b for x where
/// 'mat' is A and 'rhs' is b. The preconditioner is a function that takes
/// a residual (vector) and returns the action of the inverse preconditioner
/// on that residual.
pub fn pcg<F>(
    mat: &CsMat<f64>,
    rhs: &Array1<f64>,
    initial_iterate: &Array1<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: &F,
) -> (Array1<f64>, bool)
where
    F: Fn(&mut Array1<f64>),
{
    let mut x = initial_iterate.clone();
    let mut r = rhs - mat * &x;
    let mut r_bar = r.clone();
    preconditioner(&mut r_bar);
    let d0 = r.t().dot(&r_bar);
    let mut d = d0;
    let mut p = r_bar.clone();

    for i in 0..max_iter {
        let mut g = mat * &p;
        let alpha = d / p.t().dot(&g);
        g *= alpha;
        x += &(alpha * &p);
        r -= &g;
        r_bar = r.clone();
        preconditioner(&mut r_bar);
        let d_old = d;
        d = r.t().dot(&r_bar);

        if i % 50 == 0 {
            trace!("squared norm iter {i}: {d}");
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
