use crate::partitioner::modularity_matching;
use crate::preconditioner::{l1, multilevell1};
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use rand::{distributions::Uniform, thread_rng};

pub fn adaptive<'a>(mat: &'a CsrMatrix<f64>) -> Box<dyn Fn(&mut DVector<f64>) + 'a> {
    let mut solvers = vec![l1(mat)];
    let steps = 10;

    loop {
        let mut near_null = find_near_null(mat, &solvers);
        no_zeroes(&mut near_null);
        near_null /= near_null.norm();

        /*
        let hierarchy = modularity_matching_no_copies(
            mat.clone(),
            near_null.clone(),
            row_sums.clone(),
            inverse_total,
            2.0,
            12.0,
        );
        */
        let hierarchy = modularity_matching(mat.clone(), &near_null, 1.8);
        let ml1 = multilevell1(hierarchy);
        solvers.push(ml1);

        let convergence_rate = composite_tester(mat, steps, &solvers);
        info!(
            "components: {} convergence_rate: {}",
            solvers.len(),
            convergence_rate
        );
        if convergence_rate < 0.50 || solvers.len() == 20 {
            return build_adaptive_preconditioner(mat, solvers);
        }
    }
}

fn stationary_composite<F>(
    mat: &CsrMatrix<f64>,
    iterate: &mut DVector<f64>,
    iterations: usize,
    composite_preconditioner: &Vec<F>,
) where
    F: Fn(&mut DVector<f64>),
{
    let mut residual = -1.0 * (mat * &*iterate);
    for _ in 0..iterations {
        for component in composite_preconditioner
            .iter()
            .chain(composite_preconditioner.iter().rev())
        {
            let mut y = residual.clone();
            component(&mut y);
            *iterate += &y;
            residual -= mat * &y;
        }
    }
}

pub fn pcg_composite<'a, F>(
    mat: &'a CsrMatrix<f64>,
    x: &'a mut DVector<f64>,
    iterations: usize,
    preconditioner: F,
) where
    F: Fn(&mut DVector<f64>),
{
    let mut r = -1.0 * (mat * &*x);
    let mut r_bar = r.clone();
    preconditioner(&mut r_bar);
    let d0 = r.dot(&r_bar);
    let mut d = d0;
    let mut p = r_bar.clone();

    for _i in 0..iterations {
        let mut g = mat * &p;
        let alpha = d / p.dot(&g);
        g *= alpha;
        *x += &(alpha * &p);
        r -= &g;
        r_bar = r.clone();
        preconditioner(&mut r_bar);
        let d_old = d;
        d = r.dot(&r_bar);

        let beta = d / d_old;
        p *= beta;
        p += &r_bar;
    }
}

fn find_near_null<'a>(
    mat: &'a CsrMatrix<f64>,
    composite_preconditioner: &'a Vec<Box<dyn Fn(&mut DVector<f64>)>>,
) -> DVector<f64> {
    trace!("searching for near null");
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let mut iterate: DVector<f64> =
        DVector::from_distribution(mat.nrows(), &distribution, &mut rng);
    let mut iter = 0;

    let initial_iters = 5;
    //stationary_composite(mat, &mut iterate, initial_iters, composite_preconditioner);

    let preconditioner = |r: &mut DVector<f64>| {
        let mut x = DVector::from(vec![0.0; r.len()]);

        for component in composite_preconditioner
            .iter()
            .chain(composite_preconditioner.iter().rev())
        {
            let mut y = r.clone();
            component(&mut y);
            x += &y;
            *r -= mat * &y;
        }
        *r = x;
    };
    pcg_composite(mat, &mut iterate, initial_iters, &preconditioner);

    let mut error_norm_old = f64::MAX;

    loop {
        let error_norm_new = iterate.dot(&(mat * &iterate));

        if error_norm_new < 1e-12 {
            warn!("find_near_null converged during search");
            return iterate;
        }

        if iter % 10 == 0 && iter != 0 {
            trace!(
                "asymptotic convergence rate: {}",
                error_norm_new / error_norm_old
            );
        }

        if error_norm_new / error_norm_old > 0.99 {
            info!("convergence has stopped. Error: {:+e}", error_norm_new);
            return iterate;
        }

        //stationary_composite(mat, &mut iterate, 1, composite_preconditioner);
        pcg_composite(mat, &mut iterate, 1, &preconditioner);
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

fn composite_tester<F>(mat: &CsrMatrix<f64>, steps: usize, composite_preconditioner: &Vec<F>) -> f64
where
    F: Fn(&mut DVector<f64>),
{
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
            &crate::preconditioner::sgs(mat),
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
