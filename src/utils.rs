//! General utilities that don't have a specific home. Might move some of these
//! in with `parallel_ops` to create a `la` module.

use crate::parallel_ops::spmm_csr_dense;
use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

pub fn random_vec(size: usize) -> nalgebra::DVector<f64> {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::new(-2.0_f64, 2.0_f64);
    nalgebra::DVector::from_distribution(size, &distribution, &mut rng)
}

pub fn load_vec<P: AsRef<Path>>(path: P) -> DVector<f64> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let numbers: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        data.extend(numbers);
    }

    DVector::from(data)
}

pub fn load_boundary_dofs<P: AsRef<Path>>(path: P) -> Vec<usize> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    reader
        .lines()
        .skip(1)
        .map(|s| s.unwrap().parse().unwrap())
        .collect()
}

pub fn delete_boundary(
    dofs: Vec<usize>,
    mat: CsrMatrix<f64>,
    vec: DVector<f64>,
) -> (CsrMatrix<f64>, DVector<f64>) {
    let n = mat.nrows();
    let new_n = n - dofs.len();
    let mut p_mat = CooMatrix::new(n, new_n);

    let mut old_id = 0;
    let mut new_id = 0;
    for dof in dofs {
        while old_id < dof && old_id < n {
            p_mat.push(old_id, new_id, 1.0);
            old_id += 1;
            new_id += 1;
        }
        if old_id == n {
            break;
        }
        old_id += 1;
    }
    assert_eq!(new_id, new_n);

    let p_mat = CsrMatrix::from(&p_mat);

    let p_t = p_mat.transpose();
    let new_mat = &p_t * &(mat * &p_mat);
    let new_vec = &p_t * &vec;
    (new_mat, new_vec)
}

pub fn norm(vec: &DVector<f64>, mat: &CsrMatrix<f64>) -> f64 {
    let mut workspace = DVector::from(vec![0.0; vec.nrows()]);
    spmm_csr_dense(0.0, &mut workspace, 1.0, mat, &*vec);
    return vec.dot(&workspace).sqrt();
}

pub fn normalize(vec: &mut DVector<f64>, mat: &CsrMatrix<f64>) {
    *vec /= norm(&*vec, mat);
}
