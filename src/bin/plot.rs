use amg_match::solver::SolveInfo;
use log::trace;
use plotters::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SolveResults {
    pub num_components: usize,
    pub solve_info: SolveInfo,
}

fn main() {
    pretty_env_logger::init();
    let titles = [
        "anisotropy_2d_StationaryIteration",
        "anisotropy_2d_ConjugateGradient",
        //"spe10_StationaryIteration",
    ];
    /*
    let boomer_paths = ["images/anisotropy_2d_boomer.txt", "images/spe10_boomer.txt"];
    for (title, boomer_path) in titles.iter().zip(boomer_paths.iter()) {
        let data = load_plot_raw_data(title);
        let boomer_data = load_boomer_data(boomer_path).unwrap();
        plot_test_solve(title, &data, &boomer_data, Application::Multiplicative);
    }
    */
    for title in titles {
        let data = load_plot_raw_data(title);
        plot_test_solve_simple(title, &data);
    }
}

/*
fn load_boomer_data() -> Vec<f64> {
    let data_filename = "images/anisotropy_2d_boomer.txt";
    let file = File::open(data_filename).unwrap();
    let reader = BufReader::new(file);

    let data: Vec<f64> = reader
        .lines()
        .map(|line| line.unwrap().parse().unwrap())
        .collect();
    let r0 = data[0];
    data.into_iter().skip(1).map(|r| r / r0).collect()
}
*/

struct BoomerData {
    norms: Vec<f64>,
    operator_info: Vec<(usize, usize)>,
    complexity_measures: Vec<f64>,
    num_sweeps: usize,
}

fn load_boomer_data(path: &str) -> Result<BoomerData, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let re_iteration = Regex::new(r"Iteration : \s*\d+ \s*\|\|Br\|\| = ([\d\.]+(?:e[+-]?\d+)?)")?;
    let re_operator_matrix = Regex::new(r"^\s*\d+\s+(\d+)\s+(\d+)")?;
    let re_complexity = Regex::new(
        r"Complexity:\s*grid = (\d+\.\d+)\s*operator = (\d+\.\d+)\s*memory = (\d+\.\d+)",
    )?;
    let re_sweeps = Regex::new(r"Number of sweeps:\s*(\d+)")?;

    let mut norms = Vec::new();
    let mut operator_info = Vec::new();
    let mut complexity_measures = Vec::new();
    let mut num_sweeps = 0;

    for line in reader.lines() {
        let line = line?;

        if let Some(caps) = re_iteration.captures(&line) {
            norms.push(caps[1].parse::<f64>()?);
        }
        if let Some(caps) = re_operator_matrix.captures(&line) {
            operator_info.push((caps[1].parse::<usize>()?, caps[2].parse::<usize>()?));
        }
        if let Some(caps) = re_complexity.captures(&line) {
            complexity_measures.push(caps[1].parse::<f64>()?);
            complexity_measures.push(caps[2].parse::<f64>()?);
            complexity_measures.push(caps[3].parse::<f64>()?);
        }
        if let Some(caps) = re_sweeps.captures(&line) {
            num_sweeps = caps[1].parse::<usize>()?;
        }
    }

    let r0 = norms[0];
    norms = norms.into_iter().skip(1).map(|r| r / r0).collect();

    //println!("Norms: {:?}", norms);
    println!("Operator Info: {:?}", operator_info);
    println!("Complexity Measures: {:?}", complexity_measures);
    println!("Number of Sweeps: {}", num_sweeps);

    Ok(BoomerData {
        norms,
        operator_info,
        complexity_measures,
        num_sweeps,
    })
}

fn load_plot_raw_data(title: &str) -> Vec<SolveResults> {
    let data_filename = format!("images/{}.json", title);
    let file = File::open(data_filename).unwrap();
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).unwrap()
}

fn plot_test_solve_simple(title: &str, data: &Vec<SolveResults>) {
    let filename = format!("{}_final.png", title);
    //let filename = "temp.png";
    let root = BitMapBackend::new(&filename, (1000, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_rr_norm = 1e-12;
    let max_vcycles = 100.0;

    let data: Vec<(usize, Vec<(f64, f64)>)> = data
        .iter()
        .map(|solve_result| {
            (
                solve_result.num_components,
                solve_result
                    .solve_info
                    .relative_residual_norm_history
                    .iter()
                    .filter(|rr_norm| **rr_norm > min_rr_norm)
                    .enumerate()
                    .map(|(i, residual)| {
                        (
                            ((i + 1) * (2 * solve_result.num_components - 1)) as f64,
                            *residual,
                        )
                    })
                    .filter(|(cycles, _)| *cycles < max_vcycles)
                    //.filter(|(cycles, _)| cycles < 2000.0)
                    .collect(),
            )
        })
        .collect();

    /*
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
    */
    let plot_max = 2.0;
    let plot_min = 0.9 * min_rr_norm;

    let mut chart_context = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 100)
        .set_label_area_size(LabelAreaPosition::Bottom, 80)
        .build_cartesian_2d(0.0..max_vcycles + 1.5, (plot_min..plot_max).log_scale())
        .unwrap();

    chart_context
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

        chart_context
            .draw_series(LineSeries::new(data.iter().cloned(), color.stroke_width(2)))
            .unwrap();
        //.label(label)
        //.legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));

        let shape_style = ShapeStyle {
            color,
            filled: true,
            stroke_width: 3,
        };

        /*
        let mut step = data.len() / 10;
        if step == 0 {
            step = 1;
        }
        */
        let step = 1;
        chart_context
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

    chart_context
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

fn plot_test_solve(title: &str, data: &Vec<SolveResults>, boomer: &BoomerData) {
    let filename = format!("/{}_final.png", title);
    //let filename = "temp.png";
    let root = BitMapBackend::new(&filename, (900, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_rr_norm = 1e-12;

    let data: Vec<(usize, Vec<(f64, f64)>)> = data
        .iter()
        .map(|solve_result| {
            (
                solve_result.num_components,
                solve_result
                    .solve_info
                    .relative_residual_norm_history
                    .iter()
                    .filter(|rr_norm| **rr_norm > min_rr_norm)
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

    /*
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
    */
    let plot_max = 1.1;
    let plot_min = 0.9 * min_rr_norm;

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

    {
        let label = "BoomerAMG";
        let color = plotters::style::colors::BLACK;

        let data: Vec<(f64, f64)> = boomer
            .norms
            .iter()
            .take(*max_vcycles as usize)
            .filter(|rr_norm| **rr_norm > min_rr_norm)
            .enumerate()
            .map(|(x, y)| ((x + 1) as f64, *y))
            .collect();
        let mut step = data.len() / 10;

        chart
            .draw_series(LineSeries::new(data.iter().cloned(), color.stroke_width(2)))
            .unwrap();
        //.label(label)
        //.legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));

        let shape_style = ShapeStyle {
            color: color.into(),
            filled: true,
            stroke_width: 3,
        };

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
