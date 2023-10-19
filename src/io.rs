use nalgebra_sparse::CsrMatrix;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

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

pub fn plot_convergence(title: &str, data: &Vec<Vec<f64>>, labels: &Vec<String>) {
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
        .caption(title, ("sans-serif", 50))
        //.build_cartesian_2d(0.0..max_len as f64 + 1.5, (plot_min..plot_max).log_scale())
        .build_cartesian_2d(0.0..max_len as f64 + 1.5, plot_min..plot_max)
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .light_line_style(&WHITE.mix(0.3))
        .bold_line_style(&BLACK.mix(0.3))
        .y_desc("Asymptotic Convergence Rate (A-norm)")
        .x_desc("Components")
        .axis_desc_style(("sans-serif", 35))
        .label_style(("sans-serif", 30))
        .y_labels(8)
        .x_labels(12)
        .x_label_formatter(&|x| format!("{}", *x as usize))
        .y_label_formatter(&|y| format!("{:.1}", y))
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
