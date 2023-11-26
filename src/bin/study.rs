use std::{rc::Rc, time::Duration};

use amg_match::{
    adaptive::AdaptiveBuilder,
    io::{plot_asymptotic_convergence, plot_convergence_history, plot_convergence_history_tester},
    preconditioner::Composite,
    solver::{IterativeMethod, IterativeSolver, LogInterval, SolveInfo},
    utils::{format_duration, load_system, random_vec},
};
use nalgebra::DVector;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file as load_mm, CsrMatrix};
/*
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::Duration;
use structopt::StructOpt;
*/

#[macro_use]
extern crate log;

/*
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TestResult {
    matrix: String,
    two_level: bool,
    file: String,
    hierarchy_sizes: Vec<Vec<(usize, usize)>>, // (nrows, nnz)
    coarsening_factor: f64,
    num_unknowns: usize,
    test_iters: usize,
    construction_time: Duration,
    convergence_hist: Vec<Vec<f64>>,
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "convergence_study",
    about = "CLI for testing and comparing preconditioners"
)]
struct Opt {
    #[structopt(short, long)]
    plot: bool,
}

*/

fn main() {
    pretty_env_logger::init();

    let mfem_mats = [
        //("data/anisotropy/anisotropy_2d", "anisotropic-2d"),
        ("data/spe10/spe10_0", "spe10"),
    ];

    for (prefix, name) in mfem_mats {
        let first_name = format!("project_first/{}", name);
        study_mfem(prefix, &first_name, true);

        //let all_name = format!("project_all/{}", name);
        //study_mfem(prefix, &all_name, false);
    }

    let suitesparse_mats = ["G3_circuit", "Flan_1565"];
    for name in suitesparse_mats {
        let mat_path = format!("data/suitesparse/{}/{}.mtx", name, name);

        let first_name = format!("project_first/{}", name);
        study_suitesparse(&mat_path, &first_name, true);

        //let all_name = format!("project_all/{}", name);
        //study_suitesparse(&mat_path, &all_name, false);
    }
}

fn test_solve(name: &str, mat: Rc<CsrMatrix<f64>>, b: &DVector<f64>, pc: &mut Composite) {
    /*
    let mut step_size = (pc.components().len() / 5) - 1;
    if step_size == 0 {
        step_size = 1
    }
    */
    let step_size = 5;
    let epsilon = 1e-6;

    // components, initial_residual, residual_norms
    let mut results_pcg: Vec<(usize, SolveInfo)> = Vec::new();
    let mut results_stationary: Vec<(usize, SolveInfo)> = Vec::new();
    let max_minutes = 30;

    let dim = mat.nrows();

    info!("Solving {}", name);

    // TODO move timer inside solver?
    let _timer = std::time::Instant::now();
    let rand: DVector<f64> = random_vec(dim);

    info!("{:>15} {:>15} {:>15}", "components", "iters", "v-cycles");
    while pc.components().len() > 0 {
        let x = rand.clone();
        let pcg = IterativeSolver::new(mat.clone(), Some(x))
            .with_tolerance(epsilon)
            .with_max_iter(1000)
            .with_max_duration(Duration::from_secs(60 * max_minutes))
            .with_solver(IterativeMethod::ConjugateGradient)
            .with_preconditioner(Rc::new(pc.clone()))
            .with_log_interval(LogInterval::Time(Duration::from_secs(30)));
        let (_, solve_info_pcg) = pcg.solve(&b);

        let x = rand.clone();
        let stationary = IterativeSolver::new(mat.clone(), Some(x))
            .with_tolerance(epsilon)
            .with_max_iter(1000)
            .with_max_duration(Duration::from_secs(60 * max_minutes))
            .with_solver(IterativeMethod::StationaryIteration)
            .with_preconditioner(Rc::new(pc.clone()))
            .with_log_interval(LogInterval::Time(Duration::from_secs(30)));
        let (_, solve_info_stationary) = stationary.solve(&b);

        let components = pc.components().len();

        //TODO add solve time
        info!(
            "{:15} {:15} {:15}",
            components,
            solve_info_pcg.iterations,
            solve_info_pcg.iterations * ((2 * (components - 1)) + 1)
        );

        results_pcg.push((components, solve_info_pcg));
        results_stationary.push((components, solve_info_stationary));
        for _ in 0..step_size {
            let _ = pc.components_mut().pop();
        }

        let title_pcg = format!("{}_tester_results_pcg", name);
        let title_stationary = format!("{}_tester_results_stationary", name);
        plot_convergence_history_tester(&title_pcg, &results_pcg);
        plot_convergence_history_tester(&title_stationary, &results_stationary);
    }
}

fn study_mfem(prefix: &str, name: &str, project_first: bool) {
    let mut all_last = Vec::new();
    let mut labels = Vec::new();
    let step_size = 5;

    let coarsening_factor = 4.0;
    let (mat, b, _coords, _projector) = load_system(prefix);
    let mut adaptive_builder = AdaptiveBuilder::new(mat.clone())
        .with_max_components(45)
        .with_coarsening_factor(coarsening_factor);

    if project_first {
        adaptive_builder = adaptive_builder.with_project_first_only();
    }

    info!("Starting {} CF-{:.0}", name, coarsening_factor);
    let timer = std::time::Instant::now();
    let (mut pc, convergence_hist, _near_nulls) = adaptive_builder.build();

    /* Now that everything is Rc I can't get to the internal hierarchy anymore...
    if name == "anisotropic-2d" {
        let meshfile = "data/anisotropy/test.vtk";
        let outfile = "../skillet/error.vtk";
        write_gf(&near_nulls, &meshfile, &outfile, &projector);
        plot_hierarchy("hierarchy", &pc.components()[0].hierarchy, &coords);
    }
    */

    let construction_time = timer.elapsed();
    info!(
        "Preconitioner built in: {}",
        format_duration(&construction_time)
    );

    let min_len: usize = convergence_hist.iter().map(|vec| vec.len()).min().unwrap() - 1;
    let last: Vec<f64> = convergence_hist
        .iter()
        .map(|vec| vec[min_len].powf(1.0 / (min_len as f64)))
        .collect();
    all_last.push(last);
    let label = format!("CF: {:.0}", coarsening_factor);
    labels.push(label);

    let components_title = format!("{}_components_CF-{:.0}", name, coarsening_factor);
    plot_convergence_history(&components_title, &convergence_hist, step_size);
    let title = format!("{}_asymptotic", name);
    plot_asymptotic_convergence(&title, &all_last, &labels);
    test_solve(name, mat.clone(), &b, &mut pc);
}

/*
fn anisotropy() {
    let prefix = "data/anisotropy/anisotropy_2d";
    let name = "anisotropic-2d";
    let coarsening_factor = 8.0;
    let max_level = 16;
    let target_convergence = 0.01;
    let max_components = 20;
    let step_size = 3;
    let test_iters = None;
    let project_first_only = false;

    let mut all_last = Vec::new();
    let mut labels = Vec::new();
    let (mat, b, coords, projector) = load_system(prefix);

    info!("Starting {} CF-{:.0}", name, coarsening_factor);

    let timer = std::time::Instant::now();
    let (mut pc, convergence_hist, near_nulls) = build_adaptive(
        mat.clone(),
        coarsening_factor,
        max_level,
        target_convergence,
        max_components,
        test_iters,
        project_first_only,
    );

    let meshfile = "data/anisotropy/test.vtk";
    let outfile = "../skillet/error.vtk";
    write_gf(&near_nulls, &meshfile, &outfile, &projector);
    plot_hierarchy("hierarchy", &pc.components()[0].hierarchy, &coords);

    let construction_time = timer.elapsed();
    info!(
        "Preconitioner built in: {}",
        format_duration(&construction_time)
    );

    let max_len: usize = convergence_hist.iter().map(|vec| vec.len()).max().unwrap() - 1;
    let last: Vec<f64> = convergence_hist
        .iter()
        .map(|vec| (*vec.last().unwrap()).powf(1.0 / (vec.len() as f64)))
        .collect();

    all_last.push(last);
    let label = format!("CF: {:.0}", coarsening_factor);
    labels.push(label);

    let components_title = format!("{}_components_CF-{:.0}", name, coarsening_factor);
    plot_convergence_history(&components_title, &convergence_hist, step_size);
    let title = format!("{}_asymptotic", name);
    plot_asymptotic_convergence(&title, &all_last, &labels);
    test_solve(name, mat.clone(), &b, &mut pc);
}
fn spe10() {
    let prefix = "data/spe10/spe10_0";
    let name = "spe10";
    let coarsening_factor = 16.0;
    let max_level = 30;
    let target_convergence = 0.01;
    let max_components = 30;
    let step_size = 4;
    let test_iters = None;
    let project_first_only = false;

    let mut all_last = Vec::new();
    let mut labels = Vec::new();
    let (mat, b) = {
        let (mat, b, _coords, _projector) = load_system(prefix);
        (mat, b)
    };

    info!("Starting {} CF-{:.0}", name, coarsening_factor);

    let timer = std::time::Instant::now();
    let (mut pc, convergence_hist, _near_nulls) = build_adaptive(
        mat.clone(),
        coarsening_factor,
        max_level,
        target_convergence,
        max_components,
        test_iters,
        project_first_only,
    );

    let construction_time = timer.elapsed();
    info!(
        "Preconitioner built in: {}",
        format_duration(&construction_time)
    );

    let max_len: usize = convergence_hist.iter().map(|vec| vec.len()).max().unwrap() - 1;
    let last: Vec<f64> = convergence_hist
        .iter()
        .map(|vec| vec[max_len].powf(1.0 / (max_len as f64)))
        .collect();

    all_last.push(last);
    let label = format!("CF: {:.0}", coarsening_factor);
    labels.push(label);

    let components_title = format!("{}_components_CF-{:.0}", name, coarsening_factor);
    plot_convergence_history(&components_title, &convergence_hist, step_size);

    let title = format!("{}_asymptotic", name);
    plot_asymptotic_convergence(&title, &all_last, &labels);
    test_solve(name, mat.clone(), &b, &mut pc);
}
*/

fn study_suitesparse(mat_path: &str, name: &str, project_first: bool) {
    let step_size = 5;
    let coarsening_factor = 4.0;

    let mat = std::rc::Rc::new(CsrMatrix::from(&load_mm(mat_path).unwrap()));
    let dim = mat.nrows();
    let b: DVector<f64> = random_vec(dim);

    info!("Starting {} CF-{:.0}", name, coarsening_factor);

    let mut adaptive_builder = AdaptiveBuilder::new(mat.clone())
        .with_max_components(45)
        .with_coarsening_factor(coarsening_factor);

    if project_first {
        adaptive_builder = adaptive_builder.with_project_first_only();
    }

    info!("Starting {} CF-{:.0}", name, coarsening_factor);
    // maybe should move time inside builder
    let timer = std::time::Instant::now();
    let (mut pc, convergence_hist, _near_nulls) = adaptive_builder.build();

    let construction_time = timer.elapsed();
    info!(
        "Preconitioner built in: {}",
        format_duration(&construction_time)
    );

    let last: Vec<f64> = convergence_hist
        .iter()
        .map(|vec| (*vec.last().unwrap()).powf(1.0 / (vec.len() as f64)))
        .collect();

    let label = format!("CF: {:.0}", coarsening_factor);

    let all_last = vec![last];
    let labels = vec![label];

    let components_title = format!("{}_components_CF-{:.0}", name, coarsening_factor);
    plot_convergence_history(&components_title, &convergence_hist, step_size);

    let title = format!("{}_asymptotic", name);
    plot_asymptotic_convergence(&title, &all_last, &labels);

    test_solve(name, mat.clone(), &b, &mut pc);
}

/*
fn plot(data: &Vec<TestResult>) {
    use plotters::prelude::*;
    let spe10: Vec<TestResult> = data
        .iter()
        .filter(|test_result| test_result.matrix == SPE10.1)
        .cloned()
        .collect();
    let anisotropy: Vec<TestResult> = data
        .iter()
        .filter(|test_result| test_result.matrix == ANIS.1)
        .cloned()
        .collect();
    let split = [(spe10, "SPE10"), (anisotropy, "3d-anisotropic")];

    for (data, title) in split.iter() {
        let filename = format!("images/{}.png", title);
        let root = BitMapBackend::new(&filename, (900, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // (two_level?, coarsening_factor, scatter_data)
        let data: Vec<(bool, f64, Vec<(f64, f64)>)> = data
            .iter()
            .map(|test_result| {
                let last: Vec<(f64, f64)> = test_result
                    .convergence_hist
                    .iter()
                    .map(|vec| *vec.last().unwrap())
                    .enumerate()
                    .map(|(i, y)| (i as f64 + 1.0, y))
                    .collect();
                (test_result.two_level, test_result.coarsening_factor, last)
            })
            .collect();

        let plot_max = *data
            .iter()
            .map(|inner| inner.2.iter().map(|(_, y)| y))
            .flatten()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
            * 2.0;
        let plot_min = *data
            .iter()
            .map(|inner| inner.2.iter().map(|(_, y)| y))
            .flatten()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap()
            * 0.5;
        let max_len = data.iter().map(|data| data.2.len()).max().unwrap();

        let mut chart = ChartBuilder::on(&root)
            .set_label_area_size(LabelAreaPosition::Left, 100)
            .set_label_area_size(LabelAreaPosition::Bottom, 80)
            .caption(title, ("sans-serif", 50))
            .build_cartesian_2d(0.0..max_len as f64 + 1.5, (plot_min..plot_max).log_scale())
            .unwrap();

        chart
            .configure_mesh()
            .disable_x_mesh()
            .light_line_style(&WHITE.mix(0.3))
            .bold_line_style(&BLACK.mix(0.3))
            .y_desc("Relative Error (in A-norm, after 15 iterations)")
            .x_desc("Number of Components")
            .axis_desc_style(("sans-serif", 35))
            .label_style(("sans-serif", 30))
            .y_labels(8)
            .x_labels(12)
            .x_label_formatter(&|x| format!("{}", *x as usize))
            .y_label_formatter(&|y| format!("{:.0e}", y))
            .draw()
            .unwrap();

        for (two_level, coarsening_factor, scatter_data) in data.iter() {
            let style = ShapeStyle {
                color: Palette99::pick((*coarsening_factor as usize).ilog2() as usize).mix(1.0),
                filled: false,
                stroke_width: 3,
            };

            if *two_level {
                let label = format!("Two-Level CF: {:.0}", coarsening_factor);
                chart
                    .draw_series(
                        scatter_data
                            .iter()
                            .map(|point| Cross::new(*point, 5, style)),
                    )
                    .unwrap()
                    .label(label)
                    .legend(move |point| Cross::new(point, 5, style));
            } else {
                let label = format!("Multi-Level CF: {:.0}", coarsening_factor);
                chart
                    .draw_series(
                        scatter_data
                            .iter()
                            .map(|point| TriangleMarker::new(*point, 5, style)),
                    )
                    .unwrap()
                    .label(label)
                    .legend(move |point| TriangleMarker::new(point, 5, style));
            };
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerLeft)
            .margin(10)
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }
}
*/
