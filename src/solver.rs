use ndarray::Array1;
use sprs::{CsMat, TriMat};

pub fn pcg<F>(
    mat: &CsMat<f64>,
    rhs: &Array1<f64>,
    initial_iterate: &Array1<f64>,
    max_iter: usize,
    epsilon: f64,
    preconditioner: F,
) -> (Array1<f64>, bool)
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x = initial_iterate.clone();
    let mut r = rhs - mat * &x;
    let mut r_bar = preconditioner(&r);
    let d0 = inner_product(&r, &r_bar);
    let mut d = d0;
    let mut p = r_bar.clone();

    for i in 0..max_iter {
        let mut g = mat * &p;
        let alpha = d / inner_product(&p, &g);
        g *= alpha;
        x += &(alpha * &p);
        r -= &g;
        r_bar = preconditioner(&r);
        let d_old = d;
        d = inner_product(&r, &r_bar);

        if i % 10 == 0 {
            println!("squared norm iter {i}: {d}");
        }

        if d < epsilon * epsilon * d0 {
            println!("converged in {i} iterations");
            return (x, true);
        }

        let beta = d / d_old;
        p *= beta;
        p += &r_bar;
    }

    (x, false)
}

pub fn inner_product(lhs: &Array1<f64>, rhs: &Array1<f64>) -> f64 {
    let prod = lhs * rhs;
    prod.sum()
}
