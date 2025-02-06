use std::sync::Arc;
use std::{fs::File, io::Write, time::Duration};

use amg_match::interpolation::InterpolationType;
use amg_match::preconditioner::{LinearOperator, SmootherType, L1};
use amg_match::{
    adaptive::AdaptiveBuilder,
    preconditioner::Composite,
    solver::{Iterative, IterativeMethod, LogInterval, SolveInfo},
    utils::{format_duration, load_system},
};
use amg_match::{CsrMatrix, Vector};
use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
use ndarray::{Array, Array6};
use ndarray_linalg::krylov::{AppendResult, Orthogonalizer, MGS};
use ndarray_linalg::{InnerProduct, Norm, SVDInto};
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
        //("data/anisotropy", "anisotropic_2d"),
        //("data/spe10", "spe10_0"),
        ("data/elasticity", "elasticity_3d"),
        //("data/laplace/3d", "3d_laplace_1"),
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
    let (mat, b, _coords, rbms, projector) = load_system(prefix, name, false);
    let nrows = b.len();

    let l1 = L1::new(&mat);
    let zeros = Vector::from_elem(nrows, 0.0);
    let smoother = Iterative::new(mat.clone(), Some(zeros.clone()))
        .with_max_iter(3)
        .with_solver(IterativeMethod::StationaryIteration)
        .with_preconditioner(Arc::new(l1));

    let rbms: Vec<Vector> = rbms
        .unwrap()
        .into_iter()
        .map(|rbm| {
            let mut free = &projector * &rbm;
            smoother.apply_input(&zeros, &mut free);
            free /= free.norm();
            free
        })
        .collect();

    //let mat_ref: &CsrMatrix = mat.borrow();
    //write_mm(mat_ref, "anis_2d.mtx").expect("failed");
    let pc = study(mat, b, name);

    for (_i, rbm) in rbms.iter().enumerate() {
        for j in 0..6 {
            let ip = rbm.inner(&rbms[j]);
            print!("{:6.2} ", ip);
        }
        println!();
    }

    println!();
    for comp in pc.components() {
        let near_null = comp.get_hierarchy().get_near_null(0);
        for rbm in rbms.iter() {
            let ip = rbm.inner(&near_null);
            print!("{:6.2} ", ip.abs());
        }
        println!();
    }

    let mut mgs_rbm = MGS::new(nrows, 1e-12);
    for vec in rbms.iter() {
        match mgs_rbm.append(vec.clone()) {
            AppendResult::Added(_) => (),
            AppendResult::Dependent(_) => {
                error!("rbms are dependent...");
                panic!()
            }
        }
    }
    let rbm_q = mgs_rbm.get_q();
    let rbm_qt = rbm_q.t();

    let mut mgs_nearnull = MGS::new(nrows, 1e-12);
    for vec in pc
        .components()
        .iter()
        .map(|comp| comp.get_hierarchy().get_near_null(0))
    {
        let vec: &Vector = vec;
        match mgs_nearnull.append(vec.clone()) {
            AppendResult::Added(_) => (),
            AppendResult::Dependent(_) => {
                error!("near_nulls are dependent...");
                panic!()
            }
        }
    }
    let near_null_q = mgs_nearnull.get_q();
    let mut c = Array::zeros((6, near_null_q.ncols()));
    general_mat_mul(1.0, &rbm_qt, &near_null_q, 0.0, &mut c);
    let (_u, s, _vt) = c.svd_into(false, false).unwrap();

    let mut score = 0.0;
    trace!("SVDs of Q_rbm_t Q_nn:");
    for singular_value in s {
        print!("{:5.2} ", singular_value);
        score += singular_value;
    }
    println!();
    trace!("(1/6)*||Q_rbm* Q_nn||_2: = {:.2}", score / 6.0);

    let nn_qt = near_null_q.t();
    for (i, rbm) in rbms.iter().enumerate() {
        let mut coefs = Vector::zeros(nn_qt.nrows());
        general_mat_vec_mul(1.0, &nn_qt, &rbm, 0.0, &mut coefs);

        /*
        let mut rbm_perp = rbm.clone();
        let mut sum = 0.0;
        for (i, w) in coefs.iter().enumerate() {
            print!("{:5.2} ", w);
            sum += w * w;
            let projection = *w * near_null_q.column(i).to_owned();
            rbm_perp = rbm_perp - projection;
        }
        println!();
        trace!(
            "RBM {} ||rbm - P rbm||: {:.3}, sum of squares: {:.2}",
            i,
            rbm_perp.norm(),
            sum
        );
        */
        trace!("RBM {} ||Q_nn^T m||: {:.3}", i, coefs.norm())
    }

    /*
    //let meshfile = "data/anisotropy/test.vtk";
    //let outfile = "../skillet/error.vtk";
    //write_gf(&near_nulls, &meshfile, &outfile, &projector);
    for (i, hierarchy) in pc.components().iter().map(|ml| &ml.hierarchy).enumerate() {
        plot_hierarchy(&format!("{}_hierarchy_{}", name, i), hierarchy, &coords);
    }
    */
}

fn study_suitesparse(mat_path: &str, name: &str) {
    let mat = { Arc::new(read_matrix_market(mat_path).unwrap().to_csr()) };
    let dim = mat.rows();
    let b = Vector::random(dim, Uniform::new(-1., 1.));
    study(mat, b, name);
}

fn study(mat: Arc<CsrMatrix>, b: Vector, name: &str) -> Composite {
    info!("nrows: {} nnz: {}", mat.rows(), mat.nnz());
    let max_components = 20;
    let coarsening_factor = 7.5;
    let test_iters = 15;

    let adaptive_builder = AdaptiveBuilder::new(mat.clone())
        .with_max_components(max_components)
        .with_coarsening_factor(coarsening_factor)
        //.with_project_first_only()
        //.with_max_level(2)
        //.with_smoother(SmootherType::L1)
        .with_smoother(SmootherType::BlockL1)
        //.with_smoother(SmootherType::BlockGaussSeidel)
        .with_interpolator(InterpolationType::SmoothedAggregation((1, 0.66)))
        .with_smoothing_steps(1)
        .cycle_type(1)
        .with_max_test_iters(test_iters);

    info!("Starting {} CF-{:.0}", name, coarsening_factor);
    let timer = std::time::Instant::now();
    let (pc, _convergence_hist, _near_nulls) = adaptive_builder.build();
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

    //test_solve(name, mat.clone(), &b, pc.clone(), step_size);
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
    let max_minutes = 30;

    let dim = mat.rows();
    info!("Solving {}", name);

    //let guess: Vector = random_vec(dim).normalize();
    let guess: Vector = Vector::zeros(dim);

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
                .with_tolerance(epsilon)
                //.with_max_iter(10000)
                .with_max_duration(Duration::from_secs(60 * max_minutes))
                .with_solver(IterativeMethod::ConjugateGradient)
                .with_preconditioner(Arc::new(pc))
                .with_log_interval(LogInterval::Time(Duration::from_secs(30)))
            //.with_log_interval(LogInterval::Iterations(1))
        };
        let num_components = pc.components().len();
        let max_iter = 2000 / ((2 * (num_components - 1)) + 1);
        let solver = base_solver(mat.clone(), pc.clone(), guess.clone())
            .with_solver(method)
            .with_max_iter(max_iter);
        let complexity = pc
            .components()
            .iter()
            .map(|ml| ml.hierarchy.op_complexity())
            .sum::<f64>();
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
