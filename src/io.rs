use nalgebra_sparse::CsrMatrix;
use serde::{Deserialize, Serialize};

use crate::preconditioner::{Composite, Multilevel, PcgL1};

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

impl CompositeData {
    pub fn new(pc: &Composite<Multilevel<PcgL1>>, notes: String) -> Self {
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
