//! Some parallel implementations of basic sparse linear algebra methods
//! that are used heavily in the algorithms. These need to be fast.

use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::*;

pub fn spmm(a: &CsrMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    assert_eq!(a.ncols(), b.len());
    let mut c = DVector::<f64>::zeros(a.nrows());
    c.as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, c_i)| {
            let a_row_i = a.row(i);
            let sum = a_row_i
                .values()
                .iter()
                .zip(a_row_i.col_indices().iter())
                .fold(0.0, |acc, (val, j)| acc + b[*j] * val);
            *c_i = sum;
        });
    c
}
