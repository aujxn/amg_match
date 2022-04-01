use amg_match::partitioner::modularity;

fn main() {
    let mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/bcsstk13.mtx")
        .unwrap()
        .to_csr::<usize>();

    amg_match::mat_to_image(&mat, &format!("fine_mat.png"));
    let hierarchy = modularity(mat);

    for (i, mat) in hierarchy.matrices.iter().enumerate().take(5) {
        amg_match::mat_to_image(&mat, &format!("coarse_mat{i}.png"));
    }
}
