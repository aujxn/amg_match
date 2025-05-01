#[macro_use]
extern crate log;

use std::usize;

use faer::sparse::{Pair, SparseRowMat, SymbolicSparseRowMat, Triplet};
use faer::{mat, Mat};

fn main() {
    let a = mat![
        [1.0, 5.0, 9.0],
        [2.0, 6.0, 10.0],
        [3.0, 7.0, 11.0],
        [4.0, 8.0, 12.0f64],
    ];

    let pairs: Vec<Pair<usize, usize>> = (0..5).map(|i| Pair::new(i, i)).collect();
    let values: Vec<Mat<f64>> = (0..5).map(|_| a.clone()).collect();
    let (symbolic, _) = SymbolicSparseRowMat::try_new_from_indices(5, 5, &pairs).unwrap();

    SparseRowMat::new(symbolic, values);
}
