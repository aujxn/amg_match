use crate::partitioner::Hierarchy;
use ndarray::Array1;
use ndarray_linalg::cholesky::*;
use sprs::CsMat;

pub fn l1(mat: &CsMat<f64>) -> Box<dyn Fn(&Array1<f64>) -> Array1<f64>> {
    let l1_inverse: Array1<f64> = mat
        .outer_iterator()
        .map(|row_vec| {
            let row_sum_abs: f64 = row_vec.data().iter().map(|val| val.abs()).sum();
            1.0 / row_sum_abs
        })
        .collect();
    Box::new(move |r: &Array1<f64>| -> Array1<f64> { r * &l1_inverse })
}

pub fn multilevel(hierarchy: Hierarchy) -> Box<dyn Fn(&Array1<f64>) -> Array1<f64>> {
    let mat_coarse = hierarchy.get_matrices().last().unwrap().to_dense();
    let methods = hierarchy
        .get_matrices()
        .iter()
        .map(|mat| l1(mat))
        .collect::<Vec<_>>();
    let levels = hierarchy.get_partitions().len();

    Box::new(move |r: &Array1<f64>| -> Array1<f64> {
        let mut x_ks = Vec::with_capacity(levels);
        let mut b_ks = Vec::with_capacity(levels);
        // probably can make precon take ownership to avoid clone
        b_ks.push(r.clone());

        for (level, ((method, partition_mat), mat)) in methods
            .iter()
            .zip(hierarchy.get_partitions().iter())
            .zip(hierarchy.get_matrices().iter())
            .enumerate()
        {
            x_ks.push(method(&b_ks[level]));
            let p_t = partition_mat.transpose_view().to_owned();
            let r_k = &b_ks[level] - &(mat * &x_ks[level]);
            b_ks.push(&p_t * &r_k);
        }

        let x_c = mat_coarse.solvec(&b_ks[levels]).unwrap();
        x_ks.push(x_c);

        for level in levels - 1..=0 {
            let interpolated_x = hierarchy.get_partition(level) * &x_ks[level + 1];
            x_ks[level] += &interpolated_x;
            b_ks[level] -= &(hierarchy.get_matrix(level) * &x_ks[level]);
            x_ks[level] += &methods[level](&b_ks[level]);
        }
        x_ks.remove(0)
    })
}
