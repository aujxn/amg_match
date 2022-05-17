use crate::partitioner::modularity_matching;
use crate::preconditioner::{
    Composite, Multilevel, Preconditioner, SymmetricGaussSeidel as Sgs, L1,
};
use crate::solver::pcg;
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use rand::{distributions::Uniform, thread_rng};

pub fn build_adaptive<'a>(mat: &'a CsrMatrix<f64>) -> Composite<'a> {
    let dim = mat.nrows();
    let iterations_for_near_null = 50;
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let x: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
    let mut near_null = DVector::zeros(dim);
    let mut preconditioner = Composite::new(mat, Vec::new());
    preconditioner.push(Box::new(L1::new(mat)));

    let steps = 10;
    loop {
        let convergence_rate = composite_tester(mat, steps, &mut preconditioner);
        info!(
            "components: {} convergence_rate: {}",
            preconditioner.components().len(),
            convergence_rate
        );
        if convergence_rate < 0.85 || preconditioner.components().len() >= 3 {
            return preconditioner;
        }
        if !find_near_null(mat, &mut preconditioner, &mut near_null) {
            return preconditioner;
        }
        near_null /= near_null.norm();
        no_zeroes(&mut near_null);

        let hierarchy = modularity_matching(&mat, &near_null, 1.8);
        let ml1 = Multilevel::<L1>::new(hierarchy);
        preconditioner.push(Box::new(ml1));
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
) -> bool {
    trace!("searching for near null");
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    *near_null = DVector::from_distribution(mat.nrows(), &distribution, &mut rng);
    let mut iter = 0;

    let initial_iters = 5;
    stationary_composite(mat, near_null, initial_iters, composite_preconditioner);

    let mut error_norm_old = f64::MAX;

    loop {
        let error_norm_new = near_null.dot(&(mat * &*near_null));

        if error_norm_new < 1e-12 {
            warn!("find_near_null converged during search");
            return false;
        }

        if iter % 30 == 0 && iter != 0 {
            trace!(
                "asymptotic convergence rate: {}",
                error_norm_new / error_norm_old
            );
        }

        // TODO try difference of difference ratio
        if error_norm_new / error_norm_old > 0.997 {
            info!("convergence has stopped. Error: {:+e}", error_norm_new);
            return true;
        }

        stationary_composite(mat, near_null, 1, composite_preconditioner);
        error_norm_old = error_norm_new;
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

fn composite_tester(
    mat: &CsrMatrix<f64>,
    steps: usize,
    composite_preconditioner: &mut Composite,
) -> f64 {
    trace!("testing convergence...");
    let test_runs = 3;
    let mut avg = 0.0;
    let dim = mat.nrows();
    // 1 / (2.0 * however many steps done after test)
    let root = 1.0 / (2.0 * 3.0);
    let distribution = Uniform::new(-1e-3_f64, 1e-3_f64);
    let mut rng = thread_rng();

    for _ in 0..test_runs {
        let iterate: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
        //stationary_composite(mat, &mut iterate, steps, composite_preconditioner);
        let (mut iterate, _) = crate::solver::pcg(
            mat,
            &DVector::zeros(dim),
            &iterate,
            4,
            1e-16,
            &mut Sgs::new(mat),
        );
        let starting_error_norm = iterate.dot(&(mat * &iterate));
        stationary_composite(mat, &mut iterate, 3, composite_preconditioner);
        if starting_error_norm < 1e-16 {
            warn!("tester converged in less than number of tests ({steps})");
            return 0.0;
        }
        let final_error_norm = iterate.dot(&(mat * &iterate));
        let convergence_rate = (final_error_norm / starting_error_norm).powf(root);
        avg += convergence_rate;
    }
    avg /= test_runs as f64;

    avg
}
