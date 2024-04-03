//! General utilities that don't have a specific home. Might move some of these
//! in with `parallel_ops` to create a `la` module.

use indexmap::IndexSet;
//use crate::parallel_ops::spmm_csr_dense;
use nalgebra::DVector;
use nalgebra_sparse::{
    coo::CooMatrix, csr::CsrMatrix, io::load_coo_from_matrix_market_file as load_mm,
};
use std::{
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    rc::Rc,
    time::Duration,
};

pub fn random_vec(size: usize) -> nalgebra::DVector<f64> {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::new(-2.0_f64, 2.0_f64);
    nalgebra::DVector::from_distribution(size, &distribution, &mut rng)
}

pub fn load_system(
    prefix: &str,
) -> (
    Rc<CsrMatrix<f64>>,
    DVector<f64>,
    Vec<Vec<f64>>,
    CsrMatrix<f64>,
) {
    let matfile = format!("{}.mtx", prefix);
    let doffile = format!("{}.bdy", prefix);
    let rhsfile = format!("{}.rhs", prefix);
    let coordsfile = format!("{}.coords", prefix);

    info!("Loading linear system...");
    let mat = CsrMatrix::from(&load_mm(matfile).unwrap());

    let b = load_vec(rhsfile);
    let dofs = load_boundary_dofs(doffile);
    let mut coords = load_coords(coordsfile);

    let (mat, b, projector) = delete_boundary(dofs, mat, b, &mut coords);
    /*
    info!("Normalizing starting matrix...");
    let factor = normalize_mat(&mut mat);
    b /= factor;
    */
    (std::rc::Rc::new(mat), b, coords, projector)
    //(std::rc::Rc::new(mat), b, Vec::new(), CsrMatrix::identity(1))
}

// TODO add coords file for spe10
pub fn load_coords<P: AsRef<Path> + Display>(path: P) -> Vec<Vec<f64>> {
    let file = File::open(&path);
    match file {
        Ok(file) => BufReader::new(file)
            .lines()
            .map(|line| {
                line.unwrap()
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect()
            })
            .collect(),
        Err(_) => {
            error!("File not found: {}", path);
            // TODO make this better?
            Vec::new()
        }
    }
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
    coords: &mut Vec<Vec<f64>>,
) -> (CsrMatrix<f64>, DVector<f64>, CsrMatrix<f64>) {
    let n = mat.nrows();
    let new_n = n - dofs.len();
    let mut p_mat = CooMatrix::new(n, new_n);

    let bdy_dofs_idx: IndexSet<_> = dofs.iter().collect();
    if coords.len() > 0 {
        *coords = coords
            .iter()
            .enumerate()
            .filter(|(i, _)| !bdy_dofs_idx.contains(i))
            .map(|(_, coord)| coord)
            .cloned()
            .collect();
        assert_eq!(new_n, coords.len());
    }

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
    for _ in old_id..n {
        p_mat.push(old_id, new_id, 1.0);
        old_id += 1;
        new_id += 1;
    }
    assert_eq!(new_id, new_n);

    let p_mat = CsrMatrix::from(&p_mat);

    let p_t = p_mat.transpose();
    let new_mat = &p_t * &(mat * &p_mat);
    let new_vec = &p_t * &vec;
    (new_mat, new_vec, p_mat)
}

pub fn norm(vec: &DVector<f64>, mat: &CsrMatrix<f64>) -> f64 {
    let temp = mat * vec;
    let temp = vec.dot(&temp).sqrt();
    assert!(!temp.is_nan());
    temp
}

pub fn inner_product(
    vec_left: &DVector<f64>,
    vec_right: &DVector<f64>,
    mat: &CsrMatrix<f64>,
) -> f64 {
    //let mut workspace = DVector::from(vec![0.0; vec_right.nrows()]);
    //spmm_csr_dense(0.0, &mut workspace, 1.0, mat, &*vec_right);
    let workspace = mat * vec_right;
    return vec_left.dot(&workspace);
}

pub fn normalize(vec: &mut DVector<f64>, mat: &CsrMatrix<f64>) {
    *vec /= norm(&*vec, mat);
}

pub fn normalize_mat(mat: &mut CsrMatrix<f64>) -> f64 {
    let nrows = mat.nrows();
    let mut v = random_vec(nrows);
    let ones = DVector::from_element(nrows, 1.0);
    let mut i = 1;
    loop {
        let new_v = &*mat * &v;
        let mut eigs = new_v.component_div(&v);
        let eig = new_v.sum() / new_v.len() as f64;
        eigs -= &(eig * &ones);
        let converge = eigs.norm();
        if converge < 1e-5 {
            *mat /= eig;
            info!("Normalized after {} iterations with norm {:.2e}", i, eig);
            return eig;
        }
        if i % 1000 == 0 {
            info!(
                "Convergence on normalization: {:.2e} at iter: {}",
                converge, i
            );
        }
        v.copy_from(&new_v);
        v /= v.norm();
        i += 1;
    }
}

pub fn format_duration(duration: &Duration) -> String {
    let seconds = duration.as_secs();
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let minutes = minutes % 60;
    let seconds = seconds % 60;

    format!("{} hours, {} minutes, {} seconds", hours, minutes, seconds)
}
