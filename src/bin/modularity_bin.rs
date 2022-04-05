use amg_match::partitioner::modularity_matching;

fn main() {
    let mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/offshore.mtx")
        .unwrap()
        .to_csr::<usize>();

    //amg_match::mat_to_image(&mat, &format!("fine_mat.png"));
    let near_null = vec![1.0; mat.rows()];
    let hierarchy = modularity_matching(mat, &near_null, 2.0);

    /*
    for (i, mat) in hierarchy.get_matrices().iter().enumerate() {
        amg_match::mat_to_image(&mat, &format!("coarse_mat{i}.png"));
    }
    */
}
