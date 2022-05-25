use crate::partitioner::modularity_matching;
use crate::preconditioner::{
    Composite, Multilevel, Preconditioner, SymmetricGaussSeidel as Sgs, L1,
};
use crate::random_vec;
use crate::solver::pcg;
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use std::time::{Duration, Instant};

pub fn build_adaptive<'a>(mat: &'a CsrMatrix<f64>) -> Composite<'a> {
    let dim = mat.nrows();
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut near_null;
    let mut preconditioner = Composite::new(mat, Vec::new());
    preconditioner.push(Box::new(L1::new(mat)));
    let mut sgs = Sgs::new(mat);

    loop {
        near_null = random_vec(dim);
        let _ = pcg(mat, &zeros, &mut near_null, 10, 1e-16, &mut sgs, None);

        if let Some(convergence_rate) = find_near_null(mat, &mut preconditioner, &mut near_null) {
            near_null /= near_null.norm();
            no_zeroes(&mut near_null);

            let hierarchy = modularity_matching(&mat, &near_null, 3.5);
            let ml1 = Multilevel::<L1>::new(hierarchy);
            preconditioner.push(Box::new(ml1));
            if convergence_rate < 0.6 || preconditioner.components().len() == 15 {
                return preconditioner;
            }
        } else {
            return preconditioner;
        }
    }
}

// TODO on tester make *relative* error be convergence test
//      additive version x += 1/components (Bk^-1 * r)
//      one residual per iteration, doesnt change between components
//      try throwing early components **** dangerous
fn stationary_composite(
    mat: &CsrMatrix<f64>,
    iterate: &mut DVector<f64>,
    iterations: usize,
    composite_preconditioner: &mut Composite,
) {
    for _ in 0..iterations {
        let mut residual = -1.0 * (mat * &*iterate);
        composite_preconditioner.apply(&mut residual);
        *iterate += &residual;
    }
}

fn find_near_null<'a>(
    mat: &'a CsrMatrix<f64>,
    composite_preconditioner: &mut Composite,
    near_null: &mut DVector<f64>,
) -> Option<f64> {
    trace!("searching for near null");
    let mut iter = 0;
    let start = Instant::now();
    let mut last_log = start;
    let log_interval = Duration::from_secs(2);
    let starting_error = near_null.dot(&(mat * &*near_null));

    let mut old_convergence_rate_per_second = f64::MAX;
    let mut convergence_rate_per_second;
    let mut convergence_rate_per_iter;

    let mut stabilizer_counter = 0;

    loop {
        stationary_composite(mat, near_null, 1, composite_preconditioner);
        iter += 1;
        let current_error = near_null.dot(&(mat * &*near_null));
        if current_error < 1e-16 {
            warn!("find_near_null converged during search");
            return None;
        }

        let now = Instant::now();
        let elapsed = (now - start).as_millis();
        let convergence = current_error / starting_error;
        convergence_rate_per_second = convergence.powf(1.0 / ((elapsed as f64 / 1000.0) * 2.0));
        convergence_rate_per_iter = convergence.powf(1.0 / (iter as f64 * 2.0));

        if now - last_log > log_interval {
            trace!(
                "iteration: {} search time: {}s convergence: per second: {} per iteration: {}",
                iter,
                (now - start).as_millis() as f64 / 1000.0,
                convergence_rate_per_second,
                convergence_rate_per_iter
            );
            last_log = now;
        }

        if (convergence_rate_per_second - old_convergence_rate_per_second).abs() < 0.01 {
            stabilizer_counter += 1;
        } else {
            stabilizer_counter = 0;
        }

        if stabilizer_counter == 3 {
            info!(
                "{} components. convergence stabilized at {} on iteration {} after {} seconds. squared error: {:+e}",
                composite_preconditioner.components().len(), convergence_rate_per_iter, iter, (now - start).as_secs(), current_error
            );
            return Some(convergence_rate_per_iter);
        }

        old_convergence_rate_per_second = convergence_rate_per_second;
    }
}

fn no_zeroes(near_null: &mut DVector<f64>) {
    let mut rng = thread_rng();
    let epsilon = 1.0e-10_f64;
    let threshold = 1.0e-12_f64;
    let negative = rand::distributions::Uniform::new(-epsilon, -threshold);
    let positive = rand::distributions::Uniform::new(threshold, epsilon);

    let mut counter = 0;
    for x in near_null.iter_mut().filter(|x| x.abs() < threshold) {
        if rng.gen() {
            *x = negative.sample(&mut rng)
        } else {
            *x = positive.sample(&mut rng)
        }
        counter += 1
    }
    if counter > 0 {
        warn!("perturbed {counter} elements");
    }
}

/*
fn composite_tester(
    mat: &CsrMatrix<f64>,
    steps: usize,
    composite_preconditioner: &mut Composite,
) -> f64 {
    trace!("testing convergence...");
    let test_runs = 2;
    let mut avg_per_second = 0.0;
    let mut avg_per_iter = 0.0;
    let dim = mat.nrows();
    // 1 / (2.0 * however many steps done after test)
    let test_seconds = 5;
    let distribution = Uniform::new(-1e-3_f64, 1e-3_f64);
    let mut rng = thread_rng();

    for _ in 0..test_runs {
        let mut iterate: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
        //stationary_composite(mat, &mut iterate, steps, composite_preconditioner);
        let _ = crate::solver::pcg(
            mat,
            &DVector::zeros(dim),
            &mut iterate,
            10,
            1e-16,
            &mut Sgs::new(mat),
            None,
        );
        let start = Instant::now();
        let time = Duration::from_secs(5);
        let starting_error_norm = iterate.dot(&(mat * &iterate));
        let mut completed_iterations = 0;
        while start.elapsed() < time {
            stationary_composite(mat, &mut iterate, 1, composite_preconditioner);
            completed_iterations += 1;
        }
        if starting_error_norm < 1e-16 {
            warn!("tester converged in less than number of tests ({steps})");
            return 0.0;
        }
        let final_error_norm = iterate.dot(&(mat * &iterate));
        let convergence = final_error_norm / starting_error_norm;
        let convergence_rate_per_second = convergence.powf(1.0 / (test_seconds as f64 * 2.0));
        let convergence_rate_per_iter = convergence.powf(1.0 / (completed_iterations as f64 * 2.0));
        avg_per_second += convergence_rate_per_second;
        avg_per_iter += convergence_rate_per_iter;
    }
    avg /= test_runs as f64;

    avg
}
*/

#[cfg(test)]
mod tests {
    use crate::{adaptive::build_adaptive, preconditioner::Preconditioner, random_vec};
    use nalgebra_sparse::CsrMatrix;
    use test_generator::test_resources;

    fn test_symmetry(preconditioner: &mut dyn Preconditioner, dim: usize) {
        for _ in 0..5 {
            let u = random_vec(dim);
            let v = random_vec(dim);
            let mut preconditioned_v = v.clone();
            let mut preconditioned_u = u.clone();
            preconditioner.apply(&mut preconditioned_v);
            preconditioner.apply(&mut preconditioned_u);

            let left: f64 = u.dot(&preconditioned_v);
            let right: f64 = v.dot(&preconditioned_u);
            let difference = (left - right).abs() / (left + right).abs();
            assert!(
                difference < 1e-3,
                "\nLeft and right didn't match\nleft: {}\nright: {}\nrelative difference: {:+e}\n",
                left,
                right,
                difference
            );
        }
    }

    #[test_resources("test_matrices/unit_tests/*")]
    fn test_symmetry_adaptive(mat_path: &str) {
        let mut mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(mat_path).unwrap(),
        );

        let norm = mat
            .values()
            .iter()
            .fold(0.0_f64, |acc, x| acc + x * x)
            .sqrt();
        mat /= norm;
        let mat = mat;
        let dim = mat.nrows();
        let mut preconditioner = build_adaptive(&mat);
        test_symmetry(&mut preconditioner, dim);
    }
}
