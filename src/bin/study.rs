use amg_match::{
    adaptive::build_adaptive,
    io::plot_convergence,
    utils::{format_duration, load_system},
};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::Duration;
use structopt::StructOpt;

#[macro_use]
extern crate log;

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

static ANIS: (&'static str, &'static str) =
    ("data/anisotropy/anisotropy_3d", "anisotropic-3d-laplace");
static SPE10: (&'static str, &'static str) = ("data/spe10/spe10_0", "spe10");

fn main() {
    pretty_env_logger::init();
    let results_file = "data/out/results.json";

    let opt = Opt::from_args();
    if opt.plot {
        use std::fs::File;
        use std::io::{BufReader, Read};
        let file = File::open(results_file).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut serialized = String::new();
        buf_reader.read_to_string(&mut serialized).unwrap();
        let deserialized: Vec<TestResult> = serde_json::from_str(&serialized).unwrap();
        plot(&deserialized);
        return;
    }

    //let mats = [SPE10, ANIS];
    let mats = [ANIS];
    //let coarsening_factors = [32.0, 16.0]; //, 8.0, 4.0];
    let coarsening_factors = [16.0]; //, 8.0, 4.0];
    let max_levels = [10]; //[2, 10];
    let max_components = 20;
    let test_iters = 15;

    let mut results = Vec::new();
    let mut all_last = Vec::new();
    let mut labels = Vec::new();

    for pair in mats {
        for coarsening_factor in coarsening_factors {
            for max_level in max_levels {
                let (prefix, matrix) = pair;
                let (mat, _b) = load_system(prefix);

                let two_level = {
                    if max_level == 2 {
                        "two-level"
                    } else {
                        "multi-level"
                    }
                };
                let label = format!("{}_{}_CF-{:.0}", two_level, matrix, coarsening_factor);
                info!("Starting: {}", label);
                labels.push(label);

                let timer = std::time::Instant::now();
                let (pc, convergence_hist) = build_adaptive(
                    mat.clone(),
                    coarsening_factor,
                    max_level,
                    0.01,
                    max_components,
                    matrix,
                );

                let construction_time = timer.elapsed();
                info!(
                    "Preconitioner built in: {}",
                    format_duration(&construction_time)
                );

                let matrix = matrix.to_string();
                let file = prefix.to_string();
                let num_unknowns = mat.nrows();
                let hierarchy_sizes = pc
                    .components()
                    .iter()
                    .map(|multilevel_pc| {
                        multilevel_pc
                            .get_hierarchy()
                            .get_matrices()
                            .iter()
                            .map(|mat| (mat.nrows(), mat.nnz()))
                            .collect()
                    })
                    .collect();

                let last: Vec<f64> = convergence_hist
                    .iter()
                    .map(|vec| *vec.last().unwrap())
                    .collect();

                all_last.push(last);
                plot_convergence("Convergence", &all_last, &labels);

                let result = TestResult {
                    matrix,
                    two_level: max_level == 2,
                    file,
                    hierarchy_sizes,
                    coarsening_factor,
                    num_unknowns,
                    test_iters,
                    construction_time,
                    convergence_hist,
                };

                results.push(result);
                let serialized = serde_json::to_string(&results).unwrap();
                let mut file = std::fs::File::create(results_file).unwrap();
                file.write_all(&serialized.as_bytes()).unwrap();
            }
        }
    }
}

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
