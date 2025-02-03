//! General utilities that don't have a specific home. Might move some of these
//! in with `parallel_ops` to create a `la` module.

use indexmap::IndexSet;
//use crate::parallel_ops::spmm_csr_dense;
use nalgebra::DVector;
use nalgebra_sparse::{
    coo::CooMatrix, csr::CsrMatrix, io::load_coo_from_matrix_market_file as load_mm,
};
use std::sync::Arc;
use std::{
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    time::Duration,
};

use crate::hierarchy::Hierarchy;
use crate::interpolation::InterpolationType;
use crate::parallel_ops::spmm;
use crate::partitioner::metis_n;
use crate::preconditioner::{
    build_smoother, Identity, Multilevel, SmootherType, SymmetricGaussSeidel, L1,
};
use crate::solver::{lobpcg, Iterative, IterativeMethod};

pub fn random_vec(size: usize) -> nalgebra::DVector<f64> {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::new(-2.0_f64, 2.0_f64);
    nalgebra::DVector::from_distribution(size, &distribution, &mut rng)
}

pub fn load_system(
    prefix: &str,
    normalize_matrix: bool,
) -> (
    Arc<CsrMatrix<f64>>,
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
    let mat = mat.filter(|_, _, v| *v != 0.0);

    let b = load_vec(rhsfile).unwrap_or(DVector::from_element(mat.ncols(), 1.0).normalize());

    let dofs = load_boundary_dofs(doffile);
    let mut coords = load_coords(coordsfile);

    // TODO optional here with Option(projector) return value
    let (mut mat, mut b, projector) = delete_boundary(dofs, mat, b, &mut coords);
    if normalize_matrix {
        info!("Normalizing starting matrix...");
        let factor = normalize_mat(&mut mat);
        panic!();
        b /= factor;
    }

    (Arc::new(mat), b, coords, projector)
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

pub fn load_vec<P: AsRef<Path>>(path: P) -> Result<DVector<f64>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let numbers: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        data.extend(numbers);
    }

    Ok(DVector::from(data))
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
    vec_left.dot(&workspace)
}

pub fn normalize(vec: &mut DVector<f64>, mat: &CsrMatrix<f64>) {
    *vec /= norm(&*vec, mat);
}

pub fn normalize_mat(mat: &mut CsrMatrix<f64>) -> f64 {
    let arc_mat = Arc::new(mat.clone());
    let m = 30;
    let mut xs = (0..m)
        .into_iter()
        .map(|_| random_vec(mat.ncols()))
        .collect();
    let tol = 1e-12;

    let mut hierarchy = Hierarchy::new(arc_mat.clone());

    let mut near_null = DVector::from_element(mat.ncols(), 1.0);
    loop {
        near_null = hierarchy.add_level(
            &near_null,
            8.0,
            InterpolationType::SmoothedAggregation((1, 0.66)),
        );
        if near_null.len() < 100 {
            break;
        }
    }
    let pc = Arc::new(Multilevel::new(hierarchy, true, SmootherType::L1, 3));

    /*
    let near_null = DVector::from_element(mat.ncols(), 1.0);
    let r = metis_n(&near_null, arc_mat.clone(), 16);
    let pc = build_smoother(
        arc_mat.clone(),
        SmootherType::BlockGaussSeidel,
        r.into(),
        false,
    );
    */

    /*
        let guess = DVector::zeros(mat.ncols());
        let solver = Arc::new(
            Iterative::new(arc_mat.clone(), Some(guess))
                .with_tolerance(1e-12)
                .with_solver(IterativeMethod::ConjugateGradient)
                .with_preconditioner(pc),
        );
    */

    let max_iter = 2000;
    //let (eig, _) = lobpcg(mat, solver, x0, tol, max_iter);
    let eigs = lobpcg(mat, None, Some(pc), &mut xs, tol, max_iter);
    //let (eig, _) = lobpcg(mat, None, None, x0, tol, max_iter);

    let mut indices = (0..m).collect::<Vec<_>>();
    indices.sort_by(|idx_a, idx_b| eigs[*idx_b].partial_cmp(&eigs[*idx_a]).unwrap());
    let biggest = indices[0];
    let mat_norm: f64 = eigs[biggest];
    let eigvec = xs.swap_remove(biggest);

    trace!(
        "matrix norm: {:.2e}, eigvec norm: {:.4}, ||v||_A^2 = (v, A v) = ||A v|| = {:.4}",
        mat_norm,
        eigvec.norm(),
        eigvec.dot(&spmm(mat, &eigvec)),
    );
    *mat /= mat_norm;
    mat_norm
}

pub fn format_duration(duration: &Duration) -> String {
    let millis = duration.as_millis() % 1000;
    let seconds = duration.as_secs();
    let minutes = seconds / 60;
    let hours = minutes / 60;

    let seconds = seconds % 60;
    let minutes = minutes % 60;

    /*
        format!(
            "{} hours, {} minutes, {} seconds, {} ms",
            hours, minutes, seconds, millis
        )
    */
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}
