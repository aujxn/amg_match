use std::sync::Arc;
use std::{fs::File, io::Write, time::Duration};

use amg_match::interpolation::InterpolationType;
use amg_match::preconditioner::SmootherType;
use amg_match::{
    adaptive::AdaptiveBuilder,
    preconditioner::{Application, Composite},
    solver::{Iterative, IterativeMethod, LogInterval, SolveInfo},
    utils::{format_duration, load_system, random_vec},
};
use nalgebra::DVector;
//use nalgebra_sparse::io::save_to_matrix_market_file as write_mm;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file as load_mm, CsrMatrix};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
/*
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::Duration;
use structopt::StructOpt;
*/

/*
* TODO
* - redo cg tests with new convergence norm
* - SGS implementation
* - check spd
*
* - FOR CMIM
*   - 2 slides for interpolants (piecewise constant interpolation vs classical AMG interpolation)
*       (Show block form for classical, both recover near null (in different ways, equations),
*       operator complexity of classical for all examples (geometric complexity is size of vectors
*       CF)
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
        ("data/anisotropy/anisotropy_2d", "anisotropic-2d"),
        //("data/spe10/spe10_0", "spe10"),
        //("data/elasticity/elasticity_3d", "elasticity"),
        //("data/laplace/3d/3d_laplace_1", "laplace-3d"),
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
    let (mat, b, _coords, _projector) = load_system(prefix, true);
    //let mat_ref: &CsrMatrix<f64> = mat.borrow();
    //write_mm(mat_ref, "anis_2d.mtx").expect("failed");
    let _pc = study(mat, b, name);

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
    let mat = { Arc::new(CsrMatrix::from(&load_mm(mat_path).unwrap())) };
    let dim = mat.nrows();
    let b: DVector<f64> = random_vec(dim);
    study(mat, b, name);
}

fn study(mat: Arc<CsrMatrix<f64>>, b: DVector<f64>, name: &str) -> Composite {
    info!("nrows: {} nnz: {}", mat.nrows(), mat.nnz());
    let max_components = 20;
    let coarsening_factor = 7.5;
    let test_iters = 20;

    let adaptive_builder = AdaptiveBuilder::new(mat.clone())
        .with_max_components(max_components)
        .with_coarsening_factor(coarsening_factor)
        //.with_project_first_only()
        //.with_max_level(2)
        //.with_smoother(amg_match::preconditioner::SmootherType::L1)
        //.with_smoother(SmootherType::BlockL1)
        .with_smoother(SmootherType::BlockGaussSeidel)
        .with_interpolator(InterpolationType::SmoothedAggregation((1, 0.66)))
        .with_smoothing_steps(1)
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

    test_solve(name, mat.clone(), &b, pc.clone(), step_size);
    pc
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SolveResults {
    pub num_components: usize,
    pub solve_info: SolveInfo,
}

fn test_solve(
    name: &str,
    mat: Arc<CsrMatrix<f64>>,
    b: &DVector<f64>,
    mut pc: Composite,
    step_size: usize,
) {
    let epsilon = 1e-12;
    //let epsilon = 1e-8;
    let num_tests = 2;
    //let num_tests = 4;
    let mut all_results = vec![Vec::new(); num_tests];
    let max_minutes = 30;

    let dim = mat.nrows();
    info!("Solving {}", name);

    //let guess: DVector<f64> = random_vec(dim).normalize();
    let guess: DVector<f64> = DVector::zeros(dim);

    let log_result = |solver_type: IterativeMethod,
                      solve_results: &Vec<SolveResults>,
                      application: Application| {
        info!("{:?} Results", solver_type);
        info!("{:>15} {:>15} {:>15}", "components", "iters", "v-cycles",);
        for result in solve_results.iter() {
            info!(
                "{:15} {:15} {:15}",
                result.num_components,
                result.solve_info.iterations,
                match application {
                    Application::Multiplicative =>
                        result.solve_info.iterations * ((2 * (result.num_components - 1)) + 1),
                    Application::Random => result.solve_info.iterations,
                },
            );
        }
    };

    let solve = |method: IterativeMethod,
                 results: &mut Vec<SolveResults>,
                 application: Application,
                 pc: &mut Composite| {
        let base_solver = |mat: Arc<CsrMatrix<f64>>, pc: Composite, guess: DVector<f64>| {
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
        pc.application = application;
        let max_iter = match application {
            Application::Multiplicative => 2000 / ((2 * (num_components - 1)) + 1),
            Application::Random => 2000,
        };
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
        log_result(method, &results, application);
        let title = format!("{}_{:?}_{:?}_notonlypos", name, method, &application);
        plot_test_solve(&title, &results, application);
    };

    while pc.components().len() > 0 {
        // todo do additive?
        //let applications = [Application::Multiplicative, Application::Random];
        let applications = [Application::Multiplicative];

        let methods = [
            IterativeMethod::StationaryIteration,
            IterativeMethod::ConjugateGradient,
        ];

        //let methods = [IterativeMethod::StationaryIteration];
        let mut counter = 0;
        for method in methods.iter() {
            for application in applications.iter() {
                solve(*method, &mut all_results[counter], *application, &mut pc);
                counter += 1
            }
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

fn plot_test_solve(title: &str, data: &Vec<SolveResults>, application: Application) {
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
                    .map(|(i, residual)| match application {
                        Application::Multiplicative => (
                            ((i + 1) * ((2 * (solve_result.num_components - 1)) + 1)) as f64,
                            *residual,
                        ),
                        Application::Random => ((i + 1) as f64, *residual),
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

/*
fn plot_hierarchy(title: &str, hierarchy: &Hierarchy, coords: &Vec<Vec<f64>>) {
    let mut p = CsrMatrix::<f64>::identity(hierarchy.get_partition(0).nrows());
    for (i, new_p) in hierarchy.get_partitions().iter().enumerate() {
        p = p * new_p;
        let pt = p.transpose();
        let mut data: Vec<Vec<Vec<f64>>> = pt
            .row_iter()
            .map(|row| {
                row.col_indices()
                    .iter()
                    .map(|col_idx| coords[*col_idx].clone())
                    .collect()
            })
            .collect();

        // must make 2d in spe10 case
        if data[0][0].len() == 3 {
            data = data
                .into_iter()
                .map(|cluster| {
                    cluster
                        .into_iter()
                        .filter(|coord| (coord[2] - 100.0).abs() < 0.1)
                        .collect()
                })
                .collect();
            data = data
                .into_iter()
                .filter(|cluster| !cluster.is_empty())
                .collect();
        }

        let (x_min, x_max, y_min, y_max) = coords.iter().fold(
            (0.0, 0.0, 0.0, 0.0),
            |(mut x_min, mut x_max, mut y_min, mut y_max), coord| {
                if coord[0] < x_min {
                    x_min = coord[0];
                }
                if coord[0] > x_max {
                    x_max = coord[0];
                }
                if coord[1] < y_min {
                    y_min = coord[1];
                }
                if coord[1] > y_max {
                    y_max = coord[1];
                }
                (x_min, x_max, y_min, y_max)
            },
        );

        let filename = format!("images/{}_level{}.png", title, i);
        let root = BitMapBackend::new(&filename, (500, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            //.caption("Partition", ("sans-serif", 40))
            .build_cartesian_2d(x_min..(x_max * 1.1), y_min..(y_max * 1.05))
            .unwrap();

        ctx.configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()
            .unwrap();

        for (i, group) in data.iter().enumerate() {
            let _size = 1.5;
            let color = Palette99::pick(i).mix(1.0);
            let _cross_style = ShapeStyle {
                color,
                filled: true,
                stroke_width: 1,
            };
            ctx.draw_series(
                group
                    .iter()
                    //.map(|coord| Circle::new((coord[0], coord[1]), size, cross_style)),
                    .map(|coord| {
                        Rectangle::new(
                            [
                                (coord[0] - 10.0, coord[1] - 5.0),
                                (coord[0] + 10.0, coord[1] + 5.0),
                            ],
                            color.filled(),
                        )
                    }),
            )
            .unwrap();
        }
    }
}
*/

/*
fn write_gf(
    near_nulls: &Vec<DVector<f64>>,
    meshfile: &str,
    outfile: &str,
    projector: &CsrMatrix<f64>,
) {
    let mut contents = std::fs::read_to_string(meshfile).unwrap();

    let header: String = format!("POINT_DATA {}\n", projector.ncols());
    contents.push_str(&header);
    let near_nulls: Vec<DVector<f64>> = near_nulls
        .iter()
        .map(|near_null| projector * near_null)
        .collect();

    for (i, near_null) in near_nulls.iter().enumerate() {
        let header: String = format!("SCALARS error_vals_{} double\nLOOKUP_TABLE default\n", i);
        contents.push_str(&header);
        for val in near_null.iter() {
            contents.push_str(&format!("{}\n", val));
        }
    }

    for (i, near_null) in near_nulls.iter().enumerate() {
        let error_data: String = format!("\nVECTORS error_warp_{} double\n", i);
        contents.push_str(&error_data);
        for val in near_null.iter() {
            contents.push_str(&format!("0.0 0.0 {}\n", val));
        }
    }

    let mut file = File::create(outfile).unwrap();
    file.write_all(contents.as_bytes()).unwrap();
    trace!("writing gridfunction to mesh at: {}", outfile)
}
*/
