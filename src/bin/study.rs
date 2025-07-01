use std::sync::Arc;
use std::{fs::File, io::Write, time::Duration};

use amg_match::interpolation::InterpolationType;
use amg_match::preconditioner::{BlockSmootherType, SmootherType};
use amg_match::{
    adaptive::AdaptiveBuilder,
    preconditioner::Composite,
    solver::{Iterative, IterativeMethod, LogInterval, SolveInfo},
    utils::{format_duration, load_system},
};
use amg_match::{CsrMatrix, Vector};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use sprs::io::read_matrix_market;

/*
* TODO
* - redo cg tests with new convergence norm
* - SGS implementation
* - check spd
* - test if different smoother aggs every compononent is worth it?
*
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

/*
 * TODO / interest
 *  - Different smoothers?
 *  - serialize PC?
 *  - better documentation for publishing to crates
 *  - PAPER / SLIDES
 *      - copper mountain paper / slides!!!
 *      - paper / slides for 501 (and schedule 501 before break)
 *      - 'real' paper work
 *  - gmres?
 *  - spe10 visualization (slice of errors, coefficient, etc)
 *  - compare with boomeramg?
 *
 */

fn main() {
    pretty_env_logger::init();

    let mfem_mats = [
        //("data/anisotropy", "anisotropy_2d"),
        //("data/anisotropy", "anisotropy_3d_2r"),
        ("data/lanl/cg/", "ref4_p1"),
        //("data/spe10", "spe10_0"),
        //("data/elasticity/4", "elasticity_3d"),
        //("data/laplace/3d", "3d_laplace_1"),
        //("data/laplace", "6"),
    ];

    for (prefix, name) in mfem_mats {
        study_mfem(prefix, &name);
    }

    //let suitesparse_mats = ["G3_circuit", "Flan_1565"];
    //let suitesparse_mats = ["Flan_1565"];
    //let suitesparse_mats = ["G3_circuit"];
    /*
        let suitesparse_mats = ["boneS10", "G3_circuit", "Flan_1565"];
        for name in suitesparse_mats {
            let mat_path = format!("data/suitesparse/{}/{}.mtx", name, name);
            study_suitesparse(&mat_path, &name);
        }
    */
}

fn study_mfem(prefix: &str, name: &str) {
    let (mat, b, _coords, _rbms, _freedofs_map) = load_system(prefix, name, false);
    let b = Vector::zeros(b.dim());
    let _pc = study(mat, b, name);
}

fn study_suitesparse(mat_path: &str, name: &str) {
    let mat = { Arc::new(read_matrix_market(mat_path).unwrap().to_csr()) };
    let dim = mat.rows();
    let b = Vector::random(dim, Uniform::new(-1., 1.));
    study(mat, b, name);
}

fn study(mat: Arc<CsrMatrix>, b: Vector, name: &str) -> Composite {
    #[cfg(debug_assertions)]
    {
        let transpose = mat.transpose_view().to_csr();

        #[cfg(debug_assertions)]
        for (i, (csr_row, transposed_row)) in mat
            .outer_iterator()
            .zip(transpose.outer_iterator())
            .enumerate()
        {
            for ((j, v), (jt, vt)) in csr_row.iter().zip(transposed_row.iter()) {
                assert_eq!(j, jt);
                if vt != v {
                    let rel_err = (v - vt).abs() / v.abs().max(vt.abs());
                    let abs_err = (v - vt).abs();
                    assert!(rel_err.min(abs_err) < 1e-12, "Symmetry check failed. A_{},{} is {:.3e} but A_{},{} is {:.3e}. Relative error: {:.3e}, Absolute error: {:.3e}",
                        i, j, v, j, i, vt,
                        rel_err, abs_err);
                }
            }
        }
    }

    info!("nrows: {} nnz: {}", mat.rows(), mat.nnz());
    let max_components = 12;
    //let coarsening_factor = 16.0;
    let coarsening_factor = 8.0;
    let test_iters = 100;

    let smoother_type = SmootherType::DiagonalCompensatedBlock(
        BlockSmootherType::AutoCholesky(sprs::FillInReduction::CAMDSuiteSparse),
        //BlockSmootherType::DenseCholesky,
        //BlockSmootherType::GaussSeidel,
        //BlockSmootherType::IncompleteCholesky,
        //BlockSmootherType::ConjugateGradient(1e-12),
        1024,
        //256,
    );
    //let smoother_type = SmootherType::L1;
    //let smoother_type = SmootherType::GaussSeidel;
    let interp_type = InterpolationType::SmoothedAggregation((1, 0.66));
    //let interp_type = InterpolationType::Classical;
    let adaptive_builder = AdaptiveBuilder::new(mat.clone())
        .with_max_components(max_components)
        .with_coarsening_factor(coarsening_factor)
        .with_smoother(smoother_type)
        .with_interpolator(interp_type)
        .with_smoothing_steps(1)
        .cycle_type(1)
        //.set_block_size(3)
        .set_near_null_dim(8)
        .with_max_test_iters(test_iters);

    info!("Starting {} CF-{:.0}", name, coarsening_factor);
    let timer = std::time::Instant::now();
    let (pc, _convergence_hist, _near_nulls) = adaptive_builder.build();

    /* messing around with block smoothed aggregation...
    let kernel = hstack(&near_nulls).unwrap();
    let block_hierarchy = Hierarchy::from_partitions(
        mat.clone(),
        pc.components()[0].get_hierarchy().get_partitions().clone(),
        &kernel,
    );

    info!("Hierarchy info: {:?}", block_hierarchy);
    let smoother = SmootherType::DiagonalCompensatedBlock(BlockSmootherType::GaussSeidel, 16);
    //let smoother = SmootherType::L1;
    let sweeps = 3;
    let mu = 1;
    let ml = Multilevel::new(block_hierarchy, true, smoother, sweeps, mu);

    let epsilon = 1e-12;
    let dim = b.len();
    let guess = Vector::random(dim, Uniform::new(-1., 1.));

    let mut solver = Iterative::new(mat.clone(), Some(guess.clone()))
        .with_relative_tolerance(epsilon)
        .with_max_iter(300)
        .with_preconditioner(Arc::new(ml))
        .with_log_interval(LogInterval::Time(Duration::from_secs(10)));
    let complexity = pc.op_complexity();
    info!(
        "Starting solve with {} components and {:.2} complexity",
        pc.components().len(),
        complexity
    );

    let methods = [
        IterativeMethod::ConjugateGradient,
        IterativeMethod::StationaryIteration,
    ];

    for method in methods.iter().copied() {
        solver = solver.with_solver(method);
        solver.solve(&b);
    }
    */

    let mut step_size = pc.components().len() / 6;
    if step_size == 0 {
        step_size += 1;
    }

    let construction_time = timer.elapsed();
    info!(
        "Preconitioner built in: {}",
        format_duration(&construction_time)
    );

    //let min_len: usize = convergence_hist.iter().map(|vec| vec.len()).min().unwrap() - 1;
    /*
    let last: Vec<f64> = convergence_hist
        .iter()
        .map(|vec| vec[min_len].powf(1.0 / (min_len as f64)))
        .collect();
    */

    /*
        let components_title = format!("{}_tester_CF-{:.0}", name, coarsening_factor);
        plot_convergence_history(&components_title, &convergence_hist, step_size);
    */

    test_solve(name, mat.clone(), &b, pc.clone(), step_size);
    pc
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SolveResults {
    pub num_components: usize,
    pub solve_info: SolveInfo,
}

fn test_solve(name: &str, mat: Arc<CsrMatrix>, b: &Vector, mut pc: Composite, step_size: usize) {
    let epsilon = 1e-12;
    let num_tests = 2;
    let mut all_results = vec![Vec::new(); num_tests];
    let max_minutes = 5;

    let dim = mat.rows();
    info!("Solving {}", name);

    let guess = Vector::random(dim, Uniform::new(-1., 1.));
    //let guess: Vector = Vector::zeros(dim);

    let log_result = |solver_type: IterativeMethod, solve_results: &Vec<SolveResults>| {
        info!("{:?} Results", solver_type);
        info!("{:>15} {:>15} {:>15}", "components", "iters", "v-cycles",);
        for result in solve_results.iter() {
            info!(
                "{:15} {:15} {:15}",
                result.num_components,
                result.solve_info.iterations,
                result.solve_info.iterations * ((2 * (result.num_components - 1)) + 1),
            );
        }
    };

    let solve = |method: IterativeMethod, results: &mut Vec<SolveResults>, pc: &mut Composite| {
        let base_solver = |mat: Arc<CsrMatrix>, pc: Composite, guess: Vector| {
            Iterative::new(mat.clone(), Some(guess))
                .with_relative_tolerance(epsilon)
                .with_max_duration(Duration::from_secs(60 * max_minutes))
                .with_preconditioner(Arc::new(pc))
                .with_log_interval(LogInterval::Time(Duration::from_secs(10)))
        };
        let num_components = pc.components().len();
        let max_iter = 300 / ((2 * (num_components - 1)) + 1);
        let solver = base_solver(mat.clone(), pc.clone(), guess.clone())
            .with_solver(method)
            .with_max_iter(max_iter);
        let complexity = pc.op_complexity();
        info!(
            "Starting solve with {} components and {:.2} complexity",
            pc.components().len(),
            complexity
        );
        let (_, solve_info) = solver.solve(&b);
        results.push(SolveResults {
            num_components,
            solve_info,
        });
        log_result(method, &results);
        let title = format!("{}_{:?}", name, method);
        plot_test_solve(&title, &results);
    };

    while pc.components().len() > 0 {
        let methods = [
            IterativeMethod::StationaryIteration,
            IterativeMethod::ConjugateGradient,
        ];

        //let methods = [IterativeMethod::StationaryIteration];
        let mut counter = 0;
        for method in methods.iter() {
            solve(*method, &mut all_results[counter], &mut pc);
            counter += 1
        }

        if pc.components().len() > 1 {
            for _ in 0..step_size {
                let _ = pc.components_mut().pop();
                if pc.components().len() == 1 {
                    break;
                }
            }
        } else {
            let _ = pc.components_mut().pop();
        }
    }
}

fn save_plot_raw_data(title: &str, serialized: String) {
    let data_filename = format!("images/{}.json", title);
    trace!("Plot data filename: {}", data_filename);
    let mut file = File::create(data_filename).unwrap();
    file.write_all(&serialized.as_bytes()).unwrap();
}

fn plot_test_solve(title: &str, data: &Vec<SolveResults>) {
    let serialized = serde_json::to_string(&data).unwrap();
    save_plot_raw_data(title, serialized);

    let filename = format!("images/{}.png", title);
    let root = BitMapBackend::new(&filename, (900, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let data: Vec<(usize, Vec<(f64, f64)>)> = data
        .iter()
        .map(|solve_result| {
            (
                solve_result.num_components,
                solve_result
                    .solve_info
                    .relative_residual_norm_history
                    .iter()
                    .enumerate()
                    .map(|(i, residual)| {
                        (
                            ((i + 1) * ((2 * (solve_result.num_components - 1)) + 1)) as f64,
                            *residual,
                        )
                    })
                    //.filter(|(cycles, _)| cycles < 2000.0)
                    .collect(),
            )
        })
        .collect();

    let plot_max = *data
        .iter()
        .map(|(_, inner)| inner.iter().map(|(_, y)| y))
        .flatten()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap()
        * 1.1;
    let plot_min = *data
        .iter()
        .map(|(_, inner)| inner.iter().map(|(_, y)| y))
        .flatten()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap()
        * 0.9;

    let max_vcycles = data
        .iter()
        .map(|(_, data)| data.iter().map(|(x, _)| x))
        .flatten()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 100)
        .set_label_area_size(LabelAreaPosition::Bottom, 80)
        .build_cartesian_2d(0.0..max_vcycles + 1.5, (plot_min..plot_max).log_scale())
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .light_line_style(&WHITE.mix(0.3))
        .bold_line_style(&BLACK.mix(0.3))
        .y_desc("Relative Residual")
        .x_desc("V-Cycles")
        .axis_desc_style(("sans-serif", 35))
        .label_style(("sans-serif", 30))
        .y_labels(8)
        .x_labels(12)
        .x_label_formatter(&|x| format!("{}", *x as usize))
        .y_label_formatter(&|y| format!("{:.0e}", y))
        .draw()
        .unwrap();

    for (idx, (num_comps, data)) in data.iter().enumerate() {
        let label = format!("{} components", num_comps);
        let color = Palette99::pick(idx).mix(1.0);

        chart
            .draw_series(LineSeries::new(data.iter().cloned(), color.stroke_width(2)))
            .unwrap();
        //.label(label)
        //.legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));

        let shape_style = ShapeStyle {
            color,
            filled: true,
            stroke_width: 3,
        };

        let mut step = data.len() / 10;
        if step == 0 {
            step = 1;
        }
        chart
            .draw_series(
                data.iter()
                    .step_by(step)
                    .chain(data.iter().rev().take(1))
                    .map(|point| TriangleMarker::new(*point, 7, shape_style)),
            )
            .unwrap()
            .label(label)
            .legend(move |point| TriangleMarker::new(point, 7, shape_style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 18))
        .margin(10)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    trace!("Plotting filename: {}", filename);
}

fn plot_convergence_history(title: &str, data: &Vec<Vec<f64>>, step_size: usize) {
    let serialized = serde_json::to_string(&(data, step_size)).unwrap();
    save_plot_raw_data(title, serialized);

    let filename = format!("images/{}.png", title);
    let root = BitMapBackend::new(&filename, (900, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let plot_max = *data
        .iter()
        .step_by(step_size)
        .map(|inner| inner.iter())
        .flatten()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap()
        * 1.1;
    let plot_min = *data
        .iter()
        .step_by(step_size)
        .map(|inner| inner.iter())
        .flatten()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap()
        * 0.9;
    let data: Vec<Vec<(f64, f64)>> = data
        .iter()
        .step_by(step_size)
        .map(|inner| {
            inner
                .iter()
                .enumerate()
                .map(|(i, y)| (i as f64 + 1.0, *y))
                .collect()
        })
        .collect();
    let max_len = data.iter().map(|data| data.len()).max().unwrap();

    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 100)
        .set_label_area_size(LabelAreaPosition::Bottom, 80)
        //.caption(title, ("sans-serif", 50))
        .build_cartesian_2d(0.0..max_len as f64 + 1.5, (plot_min..plot_max).log_scale())
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .light_line_style(&WHITE.mix(0.3))
        .bold_line_style(&BLACK.mix(0.3))
        .y_desc("Relative Error")
        .x_desc("Iteration")
        .axis_desc_style(("sans-serif", 35))
        .label_style(("sans-serif", 30))
        .y_labels(8)
        .x_labels(12)
        .x_label_formatter(&|x| format!("{}", *x as usize))
        .y_label_formatter(&|y| format!("{:.0e}", y))
        .draw()
        .unwrap();

    for (idx, data) in data.iter().enumerate() {
        let color = Palette99::pick(idx).mix(1.0);
        let last_idx = data.len() - 1;
        let rho = data[last_idx].1 / data[last_idx - 1].1;
        let label = format!("{} components, ϱ = {:.2}", (idx * step_size) + 1, rho);

        chart
            .draw_series(LineSeries::new(data.iter().cloned(), color.stroke_width(2)))
            .unwrap();
        //.label(label)
        //.legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));

        let shape_style = ShapeStyle {
            color,
            filled: true,
            stroke_width: 3,
        };

        let mut step = data.len() / 10;
        if step == 0 {
            step = 1;
        }
        chart
            .draw_series(
                data.iter()
                    .step_by(step)
                    .chain(data.iter().rev().take(1))
                    .map(|point| TriangleMarker::new(*point, 7, shape_style)),
            )
            .unwrap()
            .label(label)
            .legend(move |point| TriangleMarker::new(point, 7, shape_style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerLeft)
        .label_font(("sans-serif", 18))
        .margin(10)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    trace!("Plotting filename: {}", filename);
}
