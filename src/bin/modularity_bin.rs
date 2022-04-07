use amg_match::partitioner::modularity_matching;
use amg_match::solver::pcg;
use ndarray::Array1;
use sprs::{CsMat, TriMat};

fn main() {
    //let mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/offshore.mtx")
    let mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/bcsstk13.mtx")
        .unwrap()
        .to_csr::<usize>();

    //amg_match::mat_to_image(&mat, &format!("fine_mat.png"));
    let ones = ndarray::Array::from_vec(vec![1.0; mat.rows()]);
    let zeros = ndarray::Array::from_vec(vec![0.0; mat.rows()]);
    //let _hierarchy = modularity_matching(mat, &near_null, 2.0);

    /*
    for (i, mat) in hierarchy.get_matrices().iter().enumerate() {
        amg_match::mat_to_image(&mat, &format!("coarse_mat{i}.png"));
    }
    */

    let l1_inverse: Array1<f64> = mat
        .outer_iterator()
        .map(|row_vec| {
            let row_sum_abs: f64 = row_vec.data().iter().map(|val| val.abs()).sum();
            1.0 / row_sum_abs
        })
        .collect();

    let l1_precond = |r: &Array1<f64>| -> Array1<f64> { r * &l1_inverse };

    let _x =
        pcg(&mat, &ones, &zeros, 10000, 10.0_f64.powi(-6), l1_precond).expect("didnt converge");
}
