use amg_match::{
    //mat_to_image,
    partitioner::modularity_matching,
    preconditioner::{bgs, fgs, l1, multilevelgs, multilevell1, sgs},
    solver::{pcg, stationary},
};
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use rand::{distributions::Uniform, thread_rng};
use structopt::StructOpt;
use strum_macros::{Display, EnumString};

#[macro_use]
extern crate log;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "amg_match_cli",
    about = "CLI for testing different iterative solvers and preconditioners"
)]
struct Opt {
    /// Matrix file in matrix market format
    #[structopt(parse(from_os_str))]
    input: std::path::PathBuf,

    /// Maximum number of iterations to perform
    max_iter: usize,

    /// Method to use with the solver. Options are:
    /// l1, sgs, fgs, bgs, ml1, mgs
    preconditioner: Preconditioner,

    /// Solver to use. Options are:
    /// pcg, stationary
    solver: Solver,

    /// Stop iterations after scaled residual is less
    /// than tolerance squared
    #[structopt(default_value = "10e-6")]
    tolerance: f64,
    //#[structopt(short, long)]
    //picture: Option<String>,
    //TODO: add cli for rhs option?
}

#[derive(Debug, Display, EnumString)]
#[strum(ascii_case_insensitive)]
enum Preconditioner {
    L1,
    Fgs,
    Bgs,
    Sgs,
    Ml1,
    Mgs,
    Adaptive,
}

#[derive(Debug, Display, EnumString)]
#[strum(ascii_case_insensitive)]
enum Solver {
    Pcg,
    Stationary,
}

fn main() {
    pretty_env_logger::init();
    let opt = Opt::from_args();

    let mat = CsrMatrix::from(
        &nalgebra_sparse::io::load_coo_from_matrix_market_file(&opt.input).unwrap(),
    );

    let dim = mat.nrows();
    let ones = DVector::from(vec![1.0; mat.nrows()]);
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let x: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
    let b = &mat * &x;

    let preconditioner = match opt.preconditioner {
        Preconditioner::L1 => l1(&mat),
        Preconditioner::Fgs => fgs(&mat),
        Preconditioner::Bgs => bgs(&mat),
        Preconditioner::Sgs => sgs(&mat),
        Preconditioner::Adaptive => adaptive(&mat, 5),
        _ => {
            let iterations_for_near_null = 5;
            info!(
                "calculating near null component... {} iterations using stationary L1",
                iterations_for_near_null
            );
            let (near_null, _) = stationary(
                &mat,
                &zeros,
                &ones,
                iterations_for_near_null,
                10.0_f64.powi(-6),
                &l1(&mat),
            );

            let hierarchy = modularity_matching(mat.clone(), &near_null, 2.0);
            info!(
                "Number of levels in hierarchy: {}",
                hierarchy.get_matrices().len(),
            );
            info!(
                "Size of coarsest: {}",
                hierarchy.get_matrices().last().unwrap().nrows()
            );

            match opt.preconditioner {
                Preconditioner::Ml1 => multilevell1(hierarchy),
                Preconditioner::Mgs => multilevelgs(hierarchy),
                _ => unimplemented!(),
            }
        }
    };

    let _rhs = match opt.solver {
        Solver::Stationary => stationary(
            &mat,
            &b,
            &zeros,
            opt.max_iter,
            opt.tolerance,
            &preconditioner,
        ),
        Solver::Pcg => pcg(
            &mat,
            &b,
            &zeros,
            opt.max_iter,
            opt.tolerance,
            &preconditioner,
        ),
    };
}

fn adaptive(mat: &CsrMatrix<f64>, steps: usize) -> Box<dyn Fn(&mut DVector<f64>)> {
    let mut convergence_rate = 1.0;
    let dim = mat.nrows();
    let zeros = DVector::from(vec![0.0; mat.nrows()]);
    let mut rng = thread_rng();
    let distribution = Uniform::new(-2.0_f64, 2.0_f64);
    let root = 1.0 / (steps as f64);

    let l1_preconditioner = l1(mat);
    let mut solvers = vec![];
    solvers.push(l1_preconditioner);

    while convergence_rate > 0.15 {
        let starting_iterate: DVector<f64> =
            DVector::from_distribution(dim, &distribution, &mut rng);
        let starting_residual = &zeros - mat * &starting_iterate;
        let starting_residual_norm = starting_residual.norm();

        let (near_null, _) = composite_tester(
            mat,
            &zeros,
            &starting_iterate,
            steps,
            10.0_f64.powi(-6),
            &solvers,
        );

        let final_residual = mat * &near_null - &zeros;
        let final_residual_norm = final_residual.norm();
        convergence_rate = (final_residual_norm / starting_residual_norm).powf(root);
        info!(
            "components: {} convergence_rate: {}",
            solvers.len(),
            convergence_rate
        );

        let near_null_norm = near_null.norm();
        let w = &near_null / near_null_norm;
        let hierarchy = modularity_matching(mat.clone(), &w, 2.0);
        let ml1 = multilevell1(hierarchy);
        solvers.push(ml1);
    }

    build_adaptive_preconditioner(&mat, solvers)
}

pub fn build_adaptive_preconditioner<F: 'static>(
    mat: &CsrMatrix<f64>,
    components: Vec<F>,
) -> Box<dyn Fn(&mut DVector<f64>)>
where
    F: Fn(&mut DVector<f64>),
{
    // TODO figure out lifetimes
    let mat = mat.clone();
    Box::new(move |r: &mut DVector<f64>| {
        let mut x = DVector::from(vec![0.0; r.len()]);

        for component in components.iter().chain(components.iter().rev()) {
            let mut y = r.clone();
            component(&mut y);
            x += &y;
            *r -= &mat * &y;
        }
        *r = x;
    })
}

pub fn composite_tester<F>(
    mat: &CsrMatrix<f64>,
    rhs: &DVector<f64>,
    initial_iterate: &DVector<f64>,
    iter: usize,
    _epsilon: f64,
    composite_preconditioner: &Vec<F>,
) -> (DVector<f64>, bool)
where
    F: Fn(&mut DVector<f64>),
{
    let mut residual = rhs - mat * initial_iterate;
    let mut iterate = initial_iterate.clone();

    for _i in 0..iter {
        for component in composite_preconditioner
            .iter()
            .chain(composite_preconditioner.iter().rev())
        {
            let mut y = residual.clone();
            component(&mut y);
            iterate += &y;
            residual -= mat * &y;
        }
    }
    (iterate, false)
}
