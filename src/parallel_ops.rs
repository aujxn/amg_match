use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::*;

pub fn spmm_csr_dense(
    beta: f64,
    c: &mut DVector<f64>,
    alpha: f64,
    a: &CsrMatrix<f64>,
    b: &DVector<f64>,
) {
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
            *c_i = *c_i * beta + alpha * sum;
        });
}

pub fn interpolate(fine_vec: &mut DVector<f64>, coarse_vec: &DVector<f64>, p: &CsrMatrix<f64>) {
    fine_vec
        .as_mut_slice()
        .par_iter_mut()
        .zip(p.values().par_iter().zip(p.col_indices().par_iter()))
        .for_each(|(fine, (p_val, coarse_i))| *fine = p_val * coarse_vec[*coarse_i])
}