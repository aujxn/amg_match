use crate::solver::SolveInfo;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

extern crate vtkio;

use crate::partitioner::Hierarchy;
use std::fs::File;
use std::io::Write;
/*
*
* Visualizations:
* - aggreates on 2d with several levels for anisotropic
* - spe10 aggregates
* - near null component compared to aggregates for both
* - Convergence studies:
*   - 2 level vs multi sanity_check scaling
*       - simple laplace and anisotropic
*   - for PDEs and general SPD from suitesparse
*       - multilevel asymptotic CR
*       - relative error by iteration for many components
*   - try additive version as solver (pcg only) with built PC, compare with multiplicative
*
*   ***
*   submit to student paper competition:
*   Student Competition Abstract: January 12, 2024.
*   ***
*
*/

//use crate::preconditioner::{Composite, Multilevel, L1};

#[derive(Serialize, Deserialize)]
pub struct CompositeData {
    pub hierarchies: Vec<HierarchyData>,
    pub notes: String,
}

#[derive(Serialize, Deserialize)]
pub struct HierarchyData {
    pub partition_matrices: Vec<CsrData>,
}

#[derive(Serialize, Deserialize)]
pub struct CsrData {
    num_rows: usize,
    num_cols: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

impl From<&CsrMatrix<f64>> for CsrData {
    fn from(value: &CsrMatrix<f64>) -> Self {
        CsrData {
            num_rows: value.nrows(),
            num_cols: value.ncols(),
            row_offsets: value.row_offsets().to_vec(),
            col_indices: value.col_indices().to_vec(),
            values: value.values().to_vec(),
        }
    }
}

impl Into<CsrMatrix<f64>> for CsrData {
    fn into(self) -> CsrMatrix<f64> {
        CsrMatrix::try_from_csr_data(
            self.num_rows,
            self.num_cols,
            self.row_offsets,
            self.col_indices,
            self.values,
        )
        .unwrap()
    }
}

pub fn plot_hierarchy(title: &str, hierarchy: &Hierarchy, coords: &Vec<(f64, f64)>) {
    let mut p = CsrMatrix::<f64>::identity(hierarchy.get_partition(0).nrows());
    for (i, new_p) in hierarchy.get_partitions().iter().enumerate() {
        p = p * new_p;
        let pt = p.transpose();
        let data: Vec<Vec<(f64, f64)>> = pt
            .row_iter()
            .map(|row| {
                row.col_indices()
                    .iter()
                    .map(|col_idx| coords[*col_idx])
                    .collect()
            })
            .collect();
        let (x_min, x_max, y_min, y_max) = coords.iter().fold(
            (0.0, 0.0, 0.0, 0.0),
            |(mut x_min, mut x_max, mut y_min, mut y_max), (x, y)| {
                if *x < x_min {
                    x_min = *x;
                }
                if *x > x_max {
                    x_max = *x;
                }
                if *y < y_min {
                    y_min = *y;
                }
                if *y > y_max {
                    y_max = *y;
                }
                (x_min, x_max, y_min, y_max)
            },
        );

        let filename = format!("images/{}_level{}.png", title, i);
        let root = BitMapBackend::new(&filename, (3000, 3000)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            //.caption("Partition", ("sans-serif", 40))
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();

        for (i, group) in data.iter().enumerate() {
            let size = 0.6;
            let cross_style = ShapeStyle {
                color: Palette99::pick(i).mix(1.0),
                filled: true,
                stroke_width: 1,
            };
            ctx.draw_series(
                group
                    .iter()
                    .map(|point| Circle::new(*point, size, cross_style)),
            )
            .unwrap();
        }
    }
}

pub fn write_gf(
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

pub fn plot_asymptotic_convergence(title: &str, data: &Vec<Vec<f64>>, labels: &Vec<String>) {
    let filename = format!("images/{}.png", title);
    let root = BitMapBackend::new(&filename, (900, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let plot_max = *data
        .iter()
        .map(|inner| inner.iter())
        .flatten()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap()
        * 1.1;
    let plot_min = *data
        .iter()
        .map(|inner| inner.iter())
        .flatten()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap()
        * 0.9;
    let data: Vec<Vec<(f64, f64)>> = data
        .iter()
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
        //.build_cartesian_2d(0.0..max_len as f64 + 1.5, (plot_min..plot_max).log_scale())
        .build_cartesian_2d(0.0..max_len as f64 + 1.5, plot_min..plot_max)
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .light_line_style(&WHITE.mix(0.3))
        .bold_line_style(&BLACK.mix(0.3))
        .y_desc("Convergence Factor")
        .x_desc("Components")
        .axis_desc_style(("sans-serif", 35))
        .label_style(("sans-serif", 30))
        .y_labels(8)
        .x_labels(12)
        .x_label_formatter(&|x| format!("{}", *x as usize))
        .y_label_formatter(&|y| format!("{:.2}", y))
        .draw()
        .unwrap();

    for ((idx, data), label) in data.iter().enumerate().zip(labels.iter()) {
        let cross_style = ShapeStyle {
            color: Palette99::pick(idx).mix(1.0),
            filled: false,
            stroke_width: 3,
        };
        chart
            .draw_series(data.iter().map(|point| Cross::new(*point, 5, cross_style)))
            .unwrap()
            .label(label)
            .legend(move |point| Cross::new(point, 5, cross_style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerLeft)
        .margin(10)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    trace!("Plotting filename: {}", filename);
}

pub fn plot_convergence_history(title: &str, data: &Vec<Vec<f64>>, step_size: usize) {
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
        let cross_style = ShapeStyle {
            color: Palette99::pick(idx).mix(1.0),
            filled: false,
            stroke_width: 3,
        };
        let label = format!("{} components", (idx * step_size) + 1);
        chart
            .draw_series(data.iter().map(|point| Cross::new(*point, 5, cross_style)))
            .unwrap()
            .label(label)
            .legend(move |point| Cross::new(point, 5, cross_style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerLeft)
        .margin(10)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    trace!("Plotting filename: {}", filename);
}

pub fn plot_convergence_history_tester(title: &str, data: &Vec<(usize, SolveInfo)>) {
    let filename = format!("images/{}.png", title);
    let root = BitMapBackend::new(&filename, (900, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let data: Vec<(usize, Vec<(f64, f64)>)> = data
        .iter()
        .map(|(num_components, solve_info)| {
            (
                *num_components,
                solve_info
                    .relative_residual_norm_history
                    .iter()
                    .enumerate()
                    .map(|(i, residual)| {
                        (
                            ((i + 1) * ((2 * (num_components - 1)) + 1)) as f64,
                            *residual,
                        )
                    })
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
        //.caption(title, ("sans-serif", 50))
        .build_cartesian_2d(0.0..max_vcycles + 1.5, (plot_min..plot_max).log_scale())
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .light_line_style(&WHITE.mix(0.3))
        .bold_line_style(&BLACK.mix(0.3))
        .y_desc("Relative Error")
        .x_desc("V-Cycles")
        .axis_desc_style(("sans-serif", 35))
        .label_style(("sans-serif", 30))
        .y_labels(8)
        .x_labels(12)
        .x_label_formatter(&|x| format!("{}", *x as usize))
        .y_label_formatter(&|y| format!("{:.0e}", y))
        .draw()
        .unwrap();

    for (num_comps, data) in data.iter() {
        let cross_style = ShapeStyle {
            color: Palette99::pick(*num_comps).mix(1.0),
            filled: false,
            stroke_width: 3,
        };
        let label = format!("{} components", num_comps);
        chart
            .draw_series(data.iter().map(|point| Cross::new(*point, 5, cross_style)))
            .unwrap()
            .label(label)
            .legend(move |point| Cross::new(point, 5, cross_style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerLeft)
        .margin(10)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    trace!("Plotting filename: {}", filename);
}
/*
impl CompositeData {
    pub fn new(pc: &Composite<Multilevel<L1>>, notes: String) -> Self {
        let mut hierarchies = Vec::new();

        for comp in pc.components() {
            let mut hierarchy: Vec<CsrData> = Vec::new();
            for level in comp.get_hierarchy().get_partitions() {
                hierarchy.push(level.into());
            }
            hierarchies.push(HierarchyData {
                partition_matrices: hierarchy,
            });
        }

        Self { hierarchies, notes }
    }
}
    */
