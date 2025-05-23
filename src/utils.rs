//! General utilities that don't have a specific home. Might move some of these
//! in with `parallel_ops` to create a `la` module.

use indexmap::IndexSet;
use ndarray::{Array, Array2, ArrayView2, ArrayViewMut2, Axis};
use ndarray_linalg::lobpcg::{lobpcg, LobpcgResult};
use ndarray_linalg::{EigValsh, Norm, UPLO};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use sprs::io::read_matrix_market;
use std::f64;
use std::sync::Arc;
use std::{
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    time::Duration,
};

use crate::parallel_ops::spmv;
use crate::preconditioner::{build_smoother, SmootherType};
use crate::{CooMatrix, CsrMatrix, Matrix, Vector};

pub fn load_system(
    prefix: &str,
    name: &str,
    normalize_matrix: bool,
) -> (
    Arc<CsrMatrix>,
    Vector,
    Vec<Vec<f64>>,
    Option<Vec<Vector>>,
    CsrMatrix,
) {
    let matfile = format!("{}/{}.mtx", prefix, name);
    let doffile = format!("{}/{}.bdy", prefix, name);
    let rhsfile = format!("{}/{}.rhs", prefix, name);
    let coordsfile = format!("{}/{}.coords", prefix, name);

    info!("Loading linear system...");
    let mm_mat = read_matrix_market(matfile).unwrap();
    let mut mat = CooMatrix::new((mm_mat.rows(), mm_mat.cols()));
    for (v, (i, j)) in mm_mat.triplet_iter() {
        if *v != 0.0 {
            mat.add_triplet(i, j, *v);
        }
    }
    let mat = mat.to_csr();
    info!(
        "(before bdy removal) rows: {}, nnz: {}",
        mat.rows(),
        mat.nnz()
    );

    let b = load_vec(rhsfile).unwrap_or(Vector::from_elem(mat.cols(), 1.0));
    let rbms = load_rbms(prefix).map_or(None, |rbms| Some(rbms));

    let dofs = load_boundary_dofs(doffile);
    let mut coords = load_coords(coordsfile);

    // TODO optional here with Option(projector) return value
    let (mut mat, mut b, projector) = delete_boundary(dofs, mat, b, &mut coords);
    if normalize_matrix {
        info!("Normalizing starting matrix...");
        let factor = normalize_mat(&mut mat);
        b /= factor;
    }

    (Arc::new(mat), b, coords, rbms, projector)
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

pub fn load_vec<P: AsRef<Path>>(path: P) -> Result<Vector, Box<dyn std::error::Error>> {
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

    Ok(Vector::from(data))
}

pub fn load_rbms(prefix: &str) -> Result<Vec<Vector>, Box<dyn std::error::Error>> {
    let rbm_suffixes = [
        "rbm_rotate_xy.gf",
        "rbm_rotate_yz.gf",
        "rbm_rotate_zx.gf",
        "rbm_translate_x.gf",
        "rbm_translate_y.gf",
        "rbm_translate_z.gf",
    ];
    /*
    let rbm_suffixes = [
        "rbm_translate_x.gf",
        "rbm_translate_y.gf",
        "rbm_translate_z.gf",
    ];
    */

    let mut rbms: Vec<Vector> = Vec::new();

    for suffix in rbm_suffixes {
        let rbm_path = format!("{}/{}", prefix, suffix);
        trace!("Loading RBM file: {}", rbm_path);
        let rbm_file = File::open(rbm_path)?;
        let reader = BufReader::new(rbm_file);
        let mut data = Vec::new();

        for line in reader.lines().skip(5) {
            let line = line?;
            let numbers: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            data.extend(numbers);
        }
        let mut rbm = Vector::from(data);
        rbm /= rbm.norm();

        rbms.push(rbm);
    }

    Ok(rbms)
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
    mat: CsrMatrix,
    vec: Vector,
    coords: &mut Vec<Vec<f64>>,
) -> (CsrMatrix, Vector, CsrMatrix) {
    let n = mat.rows();
    let new_n = n - dofs.len();
    let mut p_mat = CooMatrix::new((n, new_n));
    info!("DOFs length: {}, new_n: {}", dofs.len(), new_n);

    /*
    for i in 0..dofs.len() / 3 {
        let idx = i * 3;
        assert_eq!(dofs[idx] + 1, dofs[idx + 1]);
        assert_eq!(dofs[idx] + 2, dofs[idx + 2]);
    }
    */

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
            p_mat.add_triplet(old_id, new_id, 1.0);
            old_id += 1;
            new_id += 1;
        }
        if old_id == n {
            break;
        }
        old_id += 1;
    }
    for _ in old_id..n {
        p_mat.add_triplet(old_id, new_id, 1.0);
        old_id += 1;
        new_id += 1;
    }
    assert_eq!(new_id, new_n);

    let p_mat = p_mat.to_csr();

    let p_t = p_mat.transpose_view();
    let new_mat = &p_t * &(&mat * &p_mat);
    let new_vec = &p_t * &vec;

    info!(
        "(after bdy removal) rows: {}, nnz: {}",
        new_mat.rows(),
        new_mat.nnz()
    );
    (new_mat.to_csr(), new_vec, p_t.to_csr())
}

pub fn norm(vec: &Vector, mat: &CsrMatrix) -> f64 {
    let temp = mat * vec;
    let temp = vec.dot(&temp).sqrt();
    assert!(!temp.is_nan());
    temp
}

pub fn inner_product(vec_left: &Vector, vec_right: &Vector, mat: Option<&CsrMatrix>) -> f64 {
    //let mut workspace = Vector::from(vec![0.0; vec_right.rows()]);
    //spmv_csr_dense(0.0, &mut workspace, 1.0, mat, &*vec_right);
    if let Some(mat) = mat {
        let workspace = mat * vec_right;
        vec_left.dot(&workspace)
    } else {
        vec_left.dot(vec_right)
    }
}

pub fn normalize(vec: &mut Vector, mat: &CsrMatrix) {
    *vec /= norm(&*vec, mat);
}

pub fn orthonormalize_mgs(basis: &mut [Vector], mat: Option<&CsrMatrix>) {
    let n = basis.len();
    for i in 0..n {
        let norm_i = inner_product(&basis[i], &basis[i], mat).sqrt();
        if norm_i < 1e-6 {
            warn!(
                "Basis is numerically linearly dependent. norm of vector {} is {:.2e}",
                i, norm_i
            );
        }
        basis[i] /= norm_i;

        for j in i + 1..n {
            let proj = inner_product(&basis[i], &basis[j], mat);
            let scaled = proj * &basis[i];
            basis[j] -= &scaled;
        }
    }
}

pub fn normalize_mat(mat: &mut CsrMatrix) -> f64 {
    let n = mat.cols();
    let m = 10;
    let maxiter = 500;
    let xs = Array::random((n, m), Uniform::new(-1., 1.));
    let tol = 1e-3;

    let eigs = {
        let linop = |input: ArrayView2<f64>| -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((n, m));
            for (mut out_col, in_col) in out.axis_iter_mut(Axis(1)).zip(input.axis_iter(Axis(1))) {
                out_col.assign(&spmv(mat, &in_col.to_owned()));
            }
            out
        };
        let arc_mat = Arc::new(mat.clone());

        /*
        let mut near_null = Vector::from_elem(mat.cols(), 1.0);
        let mut hierarchy = Hierarchy::new(arc_mat.clone());

        loop {
            near_null = hierarchy.add_level(
                &near_null,
                8.0,
                InterpolationType::SmoothedAggregation((1, 0.66)),
                1,
            );
            if near_null.len() < 100 {
                break;
            }
        }
        let pc = Arc::new(Multilevel::new(hierarchy, true, SmootherType::L1, 3, 1));

        */
        let near_null = Arc::new(Vector::from_elem(mat.cols(), 1.0));
        let pc = build_smoother(arc_mat.clone(), SmootherType::L1, near_null, false, 1);
        let pc = |mut input: ArrayViewMut2<f64>| {
            for mut col in input.axis_iter_mut(Axis(1)) {
                let out = pc.apply(&col.to_owned());
                col.assign(&out);
            }
        };

        trace!("Starting LOBPCG to normalize matrix...");
        match lobpcg(
            linop,
            xs,
            pc,
            None,
            tol,
            maxiter,
            ndarray_linalg::TruncatedOrder::Largest,
        ) {
            LobpcgResult::Ok(vals, _vecs, _residuals) => {
                info!("LOBPCG success, eigvals: {:?}", vals);
                vals
            }
            LobpcgResult::Err(vals, _vecs, residuals, _err) => {
                info!(
                    "LOBPCG FAILED (no convergence) with eigvals: {:?} and residuals: {:?}",
                    vals, residuals
                );
                vals
            }
            LobpcgResult::NoResult(_) => panic!("LOBPCG unrecoverable failure"),
        }
        //lobpcg(linop, None, Some(pc), &mut xs, tol, max_iter)
    };

    /*
    let mut indices = (0..m).collect::<Vec<_>>();
    indices.sort_by(|idx_a, idx_b| eigs[*idx_b].partial_cmp(&eigs[*idx_a]).unwrap());
    let biggest = indices[0];
    let mat_norm: f64 = eigs[biggest];
    let eigvec = xs.swap_remove(biggest);

    trace!(
        "matrix norm: {:.2e}, eigvec norm: {:.4}, ||v||_A^2 = (v, A v) = ||A v|| = {:.4}",
        mat_norm,
        eigvec.norm(),
        eigvec.dot(&spmv(mat, &eigvec)),
    );
    */
    //*mat /= 1e-3 * eigs[0];
    //1e-3 * eigs[0]
    *mat /= eigs[0];
    eigs[0]
}

pub fn generalized_lanczos(
    c_mat: &CsrMatrix,
    b_mat: &CsrMatrix,
    start: &Vector,
    max_iter: usize,
    tol_basis: f64,
    tol_eig: f64,
) -> Vector {
    let mut alphas = Vec::new();
    let mut betas = Vec::new();
    let mut basis_vecs = Vec::new();
    let mut eigvals = Vector::zeros(1);
    let mut prev_lambda_max = f64::MAX;
    let mut delta_lambda = 0.0;

    let start_norm = norm(start, b_mat);
    basis_vecs.push(start / start_norm);

    for j in 0..max_iter {
        let vj = basis_vecs.last().unwrap();
        let uj = spmv(c_mat, &spmv(b_mat, vj));
        let z = spmv(b_mat, &uj);
        let alpha_j = vj.dot(&z);
        let wj = uj - alpha_j * vj;
        let beta_next = norm(&wj, b_mat);
        let v_next = wj / beta_next;
        basis_vecs.push(v_next);
        alphas.push(alpha_j);
        betas.push(beta_next);

        if beta_next < tol_basis {
            break;
        }
        // todo don't need to really do this...
        //orthonormalize_mgs(&mut basis_vecs, Some(b_mat));

        if j > 0 {
            let mut tri_diag = Matrix::zeros((j + 1, j + 1));
            for (i, alpha) in alphas.iter().enumerate() {
                tri_diag[[i, i]] = *alpha;
            }
            for (i, beta) in betas.iter().take(j).enumerate() {
                tri_diag[[i, i + 1]] = *beta;
                // only need upper part for eigsolver
                tri_diag[[i + 1, i]] = *beta;
            }
            // TODO should use better algorithm than this...
            // like implicit QR w/Wilkinson or Bisection + Sturm
            eigvals = tri_diag.eigvalsh(UPLO::Upper).unwrap();
            //println!("{:.2e}", eigvals);
            let lambda_max = eigvals[j];
            delta_lambda = (lambda_max - prev_lambda_max).abs();
            if delta_lambda < tol_eig {
                break;
            }
            prev_lambda_max = lambda_max;
        }
    }
    let lambda_min = eigvals[0];
    let j = eigvals.len();
    let lambda_max = eigvals[j - 1];
    info!("Lanczos result: norm of B^-1 A: {:.2e} in {} iters and (tol_basis, tol_eig) ({:.2e}, {:.2e}) with condition number: {:.2e}", lambda_max, j, betas.last().unwrap(), delta_lambda, lambda_max / lambda_min);
    trace!("{:.2e}", eigvals);
    return eigvals;
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
