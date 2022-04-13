use crate::partitioner::Hierarchy;
use ndarray::Array1;
use ndarray_linalg::cholesky::*;
use sprs::linalg::trisolve::lsolve_csr_dense_rhs as lsolve;
use sprs::linalg::trisolve::usolve_csc_dense_rhs as usolve_csc;
use sprs::linalg::trisolve::usolve_csr_dense_rhs as usolve_csr;
use sprs::CompressedStorage::CSR;
use sprs::CsMat;

pub fn l1(mat: &CsMat<f64>) -> Box<dyn Fn(&mut Array1<f64>)> {
    let l1_inverse: Array1<f64> = mat
        .outer_iterator()
        .map(|row_vec| {
            let row_sum_abs: f64 = row_vec.data().iter().map(|val| val.abs()).sum();
            1.0 / row_sum_abs
        })
        .collect();
    Box::new(move |r: &mut Array1<f64>| {
        *r *= &l1_inverse;
    })
}

pub fn fgs<'a>(mat: &'a CsMat<f64>) -> Box<dyn 'a + Fn(&mut Array1<f64>)> {
    let mut lower_triangle: CsMat<f64> = CsMat::empty(CSR, mat.rows());
    for (val, (i, j)) in mat.iter() {
        if i >= j {
            lower_triangle.insert(i, j, *val);
        }
    }
    Box::new(move |r: &mut Array1<f64>| {
        lsolve(lower_triangle.view(), r).expect("linalg error fgs");
    })
}

pub fn bgs<'a>(mat: &'a CsMat<f64>) -> Box<dyn 'a + Fn(&mut Array1<f64>)> {
    let mut upper_triangle: CsMat<f64> = CsMat::empty(CSR, mat.rows());
    for (val, (i, j)) in mat.iter() {
        if i <= j {
            upper_triangle.insert(i, j, *val);
        }
    }
    Box::new(move |r: &mut Array1<f64>| {
        usolve_csr(upper_triangle.view(), r).expect("linalg error fgs");
    })
}

pub fn sgs<'a>(mat: &'a CsMat<f64>) -> Box<dyn 'a + Fn(&mut Array1<f64>)> {
    let diag: Array1<f64> = mat
        .diag_iter()
        .map(|x| *x.expect("missing diagonal element"))
        .collect();
    let mut lower_triangle: CsMat<f64> = CsMat::empty(CSR, mat.rows());
    for (val, (i, j)) in mat.iter() {
        if i >= j {
            lower_triangle.insert(i, j, *val);
        }
    }
    Box::new(move |r: &mut Array1<f64>| {
        lsolve(lower_triangle.view(), r.view_mut()).expect("lsolve sgs linalg error");
        *r *= &diag;
        usolve_csc(lower_triangle.transpose_view(), r).expect("usolve sgs linalg error");
    })
}

pub fn multilevel(hierarchy: Hierarchy) -> Box<dyn Fn(&mut Array1<f64>)> {
    let mat_coarse = hierarchy.get_matrices().last().unwrap().to_dense();
    let methods = hierarchy
        .get_matrices()
        .iter()
        .map(|mat| l1(mat))
        .collect::<Vec<_>>();
    let levels = hierarchy.get_partitions().len();

    Box::new(move |r: &mut Array1<f64>| {
        let mut x_ks = Vec::with_capacity(levels);
        let mut b_ks = Vec::with_capacity(levels);
        // probably can make precon take ownership to avoid clone
        b_ks.push(r.clone());

        for (level, ((method, partition_mat), mat)) in methods
            .iter()
            .zip(hierarchy.get_partitions().iter())
            .zip(hierarchy.get_matrices().iter())
            .enumerate()
        {
            x_ks.push(b_ks[level].clone());
            method(x_ks.last_mut().unwrap());
            let p_t = partition_mat.transpose_view().to_owned();
            let r_k = &b_ks[level] - &(mat * &x_ks[level]);
            b_ks.push(&p_t * &r_k);
        }

        let x_c = mat_coarse.solvec(&b_ks[levels]).unwrap();
        x_ks.push(x_c);

        for level in (0..levels).rev() {
            let interpolated_x = hierarchy.get_partition(level) * &x_ks[level + 1];
            x_ks[level] += &interpolated_x;
            b_ks[level] -= &(hierarchy.get_matrix(level) * &x_ks[level]);
            methods[level](&mut b_ks[level]);
            x_ks[level] += &b_ks[level];
        }
        *r = x_ks.remove(0);
    })
}
