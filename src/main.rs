use amg_match::luby::lubys;

fn main() {
    let mut mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/bcsstk13.mtx")
        .unwrap()
        .to_csr::<usize>();
    let _n = mat.rows();

    let image_as_2darray = sprs::visu::nnz_image(mat.view());
    let (h, w) = image_as_2darray.dim();
    let raw_image = image_as_2darray.into_raw_vec();
    let png = image::GrayImage::from_raw(w as u32, h as u32, raw_image).unwrap();
    png.save("out.png").unwrap();

    let row_sums: Vec<f64> = mat
        .outer_iterator()
        .map(|row| row.data().iter().sum())
        .collect();

    let total_sum: f64 = row_sums.iter().sum();
    println!("{}", total_sum);

    for i in 0..10 {
        let partition_mat = lubys(&mat, None);
        mat = &partition_mat.transpose_view().to_owned() * &(&mat * &partition_mat);
        if i % 2 == 0 {
            let i = i / 2;
            amg_match::mat_to_image(&mat, &format!("coarse_mat{i}.png"));
        }
    }
}
