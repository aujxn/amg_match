use amg_match::partitioner::lubys;

fn main() {
    let mut mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/bcsstk13.mtx")
        .unwrap()
        .to_csr::<usize>();
    amg_match::mat_to_image(&mat, "fine_mat.png");

    for i in 0..10 {
        let partition_mat = lubys(&mat, None);
        mat = &partition_mat.transpose_view().to_owned() * &(&mat * &partition_mat);
        if i % 2 == 0 {
            let i = i / 2;
            amg_match::mat_to_image(&mat, &format!("coarse_mat{i}.png"));
        }
    }
}
