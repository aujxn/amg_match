use amg_match::partitioner::modularity_matching;
use amg_match::preconditioner::{l1, multilevel};
use amg_match::solver::pcg;
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    //let mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/offshore.mtx")
    //let mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/bcsstk13.mtx")
    let mat = sprs::io::read_matrix_market::<f64, usize, _>("test_matrices/1024.mtx")
        .unwrap()
        .to_csr::<usize>();
    let dim = mat.rows();

    //amg_match::mat_to_image(&mat, &format!("fine_mat.png"));
    let ones = ndarray::Array::from_vec(vec![1.0; mat.rows()]);
    let zeros = ndarray::Array::from_vec(vec![0.0; mat.rows()]);

    let x: Array1<f64> = ndarray::Array::random(dim, Uniform::new(-2., 2.));
    let b = &mat * &x;

    println!("pcg with l1 as preconditioner");
    let l1_precond = l1(&mat);
    let _ = pcg(&mat, &b, &zeros, 10000, 10.0_f64.powi(-6), l1_precond);

    let l1_precond = l1(&mat);
    let (near_null, _) = pcg(&mat, &zeros, &ones, 10, 10.0_f64.powi(-6), l1_precond);
    let hierarchy = modularity_matching(mat.clone(), &near_null, 2.0);
    println!(
        "Number of levels in hierarchy: {}",
        hierarchy.get_matrices().len()
    );
    let multilevel_preconditioner = multilevel(hierarchy);

    /* symmetry check --- seems good */

    /*
    for _ in 0..50 {
        let u = ndarray::Array::random(dim, Uniform::new(0., 2.));
        let v = ndarray::Array::random(dim, Uniform::new(0., 2.));
        let left: f64 = (&u * &multilevel_preconditioner(&v)).iter().sum();
        let right: f64 = (&v * &multilevel_preconditioner(&u)).iter().sum();
        assert!(left - right < 10.0_f64.powi(-6));
        let pos: f64 = (&u * &multilevel_preconditioner(&u)).iter().sum();
        assert!(pos > 0.0);
        let pos: f64 = (&v * &multilevel_preconditioner(&v)).iter().sum();
        assert!(pos > 0.0);
    }
    */

    /*
    for (i, mat) in hierarchy.get_matrices().iter().enumerate() {
        amg_match::mat_to_image(&mat, &format!("coarse_mat{i}.png"));
    }
    */

    println!("pcg with multilevel as preconditioner");
    let _x = pcg(
        &mat,
        &b,
        &zeros,
        10000,
        10.0_f64.powi(-6),
        multilevel_preconditioner,
    );
}
