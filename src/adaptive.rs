use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use rand::{distributions::Uniform, thread_rng};

pub fn find_near_null<F>(
    mat: &CsrMatrix<f64>,
    preconditioner: &F,
) -> (DVector<f64>, DVector<f64>, f64)
where
    F: Fn(&mut DVector<f64>),
{
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let mut near_null: DVector<f64> =
        DVector::from_distribution(mat.nrows(), &distribution, &mut rng);

    loop {
        no_zeroes(&mut near_null);
        let near_null = tester(mat, preconditioner);
        result /= result.norm();

        if let Some((row_sums, inverse_total)) = try_row_sums(mat, &mut result) {
            return (result, row_sums, inverse_total);
        }
        if converged {
            near_null = DVector::from_distribution(mat.nrows(), &distribution, &mut rng);
        } else {
            near_null = result;
        }
    }
}

fn tester<F>(mat: &CsrMatrix<f64>, preconditioner: &F) -> DVector<f64>
where
    F: Fn(&mut DVector<f64>),
{
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let mut x: DVector<f64> = DVector::from_distribution(mat.nrows(), &distribution, &mut rng);
    let mut r = -1.0 * &(mat * &x);
    let mut iter = 0;

    let initial_iters = 5;
    for _ in 0..initial_iters {
        preconditioner(&mut r);
        x += &r;
    }
    let mut r_norm_old = r.norm();

    loop {
        r = -1.0 * &(mat * &x);
        let r_norm_new = r.dot(&r);

        if iter % 10 == 0 {
            trace!("residual norm on iter {iter}: {r_norm_new}");
        }

        if r_norm_new / r_norm_old > 0.999 {
            info!("convergence has stopped");
            return x;
        }

        preconditioner(&mut r);
        x += &r;
        r_norm_old = r_norm_new;
        iter += 1;
    }
}

fn no_zeroes(near_null: &mut DVector<f64>) {
    let mut rng = thread_rng();
    let epsilon = 1.0e-10_f64;
    let threshold = 1.0e-12_f64;
    let negative = rand::distributions::Uniform::new(-epsilon, -threshold);
    let positive = rand::distributions::Uniform::new(threshold, epsilon);

    for x in near_null.iter_mut().filter(|x| x.abs() < threshold) {
        warn!("perturbed element");
        if rng.gen() {
            *x = negative.sample(&mut rng)
        } else {
            *x = positive.sample(&mut rng)
        }
    }
}
