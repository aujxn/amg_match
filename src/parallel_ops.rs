//! Some parallel implementations of basic sparse linear algebra methods
//! that are used heavily in the algorithms. These need to be fast.

use rayon::prelude::*;

use crate::{CsrMatrix, Vector};

pub fn spmv(a: &CsrMatrix, b: &Vector) -> Vector {
    assert!(a.is_csr());
    assert_eq!(a.cols(), b.len());
    let c: Vec<f64> = (0..a.rows())
        .into_par_iter()
        .map(|i| {
            let row = a.outer_view(i).unwrap();
            row.iter().map(|(j, val)| b[j] * val).sum::<f64>()
        })
        .collect();
    Vector::from(c)
}
