use crate::partitioner::modularity_matching;
use crate::preconditioner_new::{Multilevel, Preconditioner, L1};
use crate::solver::pcg;
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use rand::{distributions::Uniform, thread_rng};

pub struct Adaptive<'a> {
    mat: &'a CsrMatrix<f64>,
    components: Vec<Multilevel<L1>>,
}

impl<'a> Preconditioner for Adaptive<'a> {
    fn apply(&mut self, r: &mut DVector<f64>) {
        let mut x = DVector::from(vec![0.0; r.len()]);

        for component in self.components.iter_mut() {
            let mut y = r.clone();
            component.apply(&mut y);
            x += &y;
            *r -= self.mat * &y;
        }
        for component in self.components.iter_mut().rev() {
            let mut y = r.clone();
            component.apply(&mut y);
            x += &y;
            *r -= self.mat * &y;
        }
        *r = x;
    }
}

impl<'a> Adaptive<'a> {
    pub fn new(mat: &'a CsrMatrix<f64>) -> Self {
        let dim = mat.nrows();
        let iterations_for_near_null = 50;
        let zeros = DVector::from(vec![0.0; mat.nrows()]);
        let mut rng = thread_rng();
        let distribution = Uniform::new(-2.0_f64, 2.0_f64);
        let x: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);

        let (near_null, _) = pcg(
            mat,
            &zeros,
            &x,
            iterations_for_near_null,
            10.0_f64.powi(-6),
            &mut L1::new(&mat),
        );

        let hierarchy = modularity_matching(mat.clone(), &near_null, 2.0);
        let mut components = vec![Multilevel::<L1>::new(hierarchy)];

        let steps = 10;
        loop {
            match find_near_null(mat, &mut components) {
                Some(mut near_null) => {
                    no_zeroes(&mut near_null);
                    near_null /= near_null.norm();

                    let hierarchy = modularity_matching(mat.clone(), &near_null, 1.8);
                    let ml1 = Multilevel::<L1>::new(hierarchy);
                    components.push(ml1);

                    let convergence_rate = composite_tester(mat, steps, &mut components);
                    info!(
                        "components: {} convergence_rate: {}",
                        components.len(),
                        convergence_rate
                    );
                    if convergence_rate < 0.50 || components.len() == 20 {
                        return Self { mat, components };
                    }
                }
                None => return Self { mat, components },
            }
        }
    }
}

fn stationary_composite(
    mat: &CsrMatrix<f64>,
    iterate: &mut DVector<f64>,
    iterations: usize,
    composite_preconditioner: &mut Vec<Multilevel<L1>>,
) {
    let mut residual = -1.0 * (mat * &*iterate);
    for _ in 0..iterations {
        for component in composite_preconditioner.iter_mut() {
            let mut y = residual.clone();
            component.apply(&mut y);
            *iterate += &y;
            residual -= mat * &y;
        }
        for component in composite_preconditioner.iter_mut().rev() {
            let mut y = residual.clone();
            component.apply(&mut y);
            *iterate += &y;
            residual -= mat * &y;
        }
    }
}

fn find_near_null<'a>(
    mat: &'a CsrMatrix<f64>,
    composite_preconditioner: &mut Vec<Multilevel<L1>>,
) -> Option<DVector<f64>> {
    trace!("searching for near null");
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let mut iterate: DVector<f64> =
        DVector::from_distribution(mat.nrows(), &distribution, &mut rng);
    let mut iter = 0;

    let initial_iters = 5;
    stationary_composite(mat, &mut iterate, initial_iters, composite_preconditioner);

    let mut error_norm_old = f64::MAX;

    loop {
        let error_norm_new = iterate.dot(&(mat * &iterate));

        if error_norm_new < 1e-12 {
            warn!("find_near_null converged during search");
            return None;
        }

        if iter % 10 == 0 && iter != 0 {
            trace!(
                "asymptotic convergence rate: {}",
                error_norm_new / error_norm_old
            );
        }

        if error_norm_new / error_norm_old > 0.97 {
            info!("convergence has stopped. Error: {:+e}", error_norm_new);
            return Some(iterate);
        }

        stationary_composite(mat, &mut iterate, 1, composite_preconditioner);
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

pub fn build_adaptive_preconditioner<'a>(
    mat: &'a CsrMatrix<f64>,
    components: Vec<Box<dyn Fn(&mut DVector<f64>)>>,
) -> Box<dyn Fn(&mut DVector<f64>) + 'a> {
    Box::new(move |r: &mut DVector<f64>| {
        let mut x = DVector::from(vec![0.0; r.len()]);

        for component in components.iter().chain(components.iter().rev()) {
            let mut y = r.clone();
            component(&mut y);
            x += &y;
            *r -= mat * &y;
        }
        *r = x;
    })
}

fn composite_tester(
    mat: &CsrMatrix<f64>,
    steps: usize,
    composite_preconditioner: &mut Vec<Multilevel<L1>>,
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
            &mut crate::preconditioner_new::SymmetricGaussSeidel::new(mat),
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
