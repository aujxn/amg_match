use crate::partitioner::Hierarchy;
use ndarray::{Array, Array1};
use ndarray_linalg::cholesky::*;
use sprs::{
    linalg::trisolve::{
        lsolve_csr_dense_rhs as lsolve, usolve_csc_dense_rhs as usolve_csc,
        usolve_csr_dense_rhs as usolve_csr,
    },
    CompressedStorage::CSR,
    CsMat,
};

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

pub fn fgs(mat: &CsMat<f64>) -> Box<dyn Fn(&mut Array1<f64>)> {
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

pub fn bgs(mat: &CsMat<f64>) -> Box<dyn Fn(&mut Array1<f64>)> {
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

pub fn multilevelgs(hierarchy: Hierarchy) -> Box<dyn Fn(&mut Array1<f64>)> {
    let mat_coarse = hierarchy.get_matrices().last().unwrap().to_dense();
    let presmooth = hierarchy
        .get_matrices()
        .iter()
        .map(|mat| fgs(mat))
        .collect::<Vec<_>>();
    let postsmooth = hierarchy
        .get_matrices()
        .iter()
        .map(|mat| bgs(mat))
        .collect::<Vec<_>>();
    let levels = hierarchy.get_partitions().len();

    Box::new(move |r: &mut Array1<f64>| {
        let mut x_ks = Vec::with_capacity(levels);
        let mut b_ks = Vec::with_capacity(levels);
        b_ks.push(r.clone());

        for (level, ((presmoother, partition_mat), mat)) in presmooth
            .iter()
            .zip(hierarchy.get_partitions().iter())
            .zip(hierarchy.get_matrices().iter())
            .enumerate()
        {
            x_ks.push(b_ks[level].clone());
            presmoother(x_ks.last_mut().unwrap());
            let p_t = partition_mat.transpose_view().to_owned();
            let r_k = &b_ks[level] - &(mat * &x_ks[level]);
            b_ks.push(&p_t * &r_k);
        }

        let x_c = mat_coarse.solvec(&b_ks[levels]).unwrap();
        x_ks.push(x_c);

        for (level, (((partition_mat, postsmoother), b_k), mat)) in hierarchy
            .get_partitions()
            .iter()
            .zip(postsmooth.iter())
            .zip(b_ks.iter_mut())
            .zip(hierarchy.get_matrices().iter())
            .enumerate()
            .rev()
        {
            let interpolated_x = partition_mat * &x_ks[level + 1];
            x_ks[level] += &interpolated_x;
            *b_k -= &(mat * &x_ks[level]);
            postsmoother(b_k);
            x_ks[level] += &*b_k;
        }
        *r = x_ks.remove(0);
    })
}

pub fn multilevell1(hierarchy: Hierarchy) -> Box<dyn Fn(&mut Array1<f64>)> {
    let mat_coarse = hierarchy.get_matrices().last().unwrap().to_dense();
    let smoothing_steps = 10;
    let smoothers = hierarchy
        .get_matrices()
        .iter()
        .map(|mat| l1(mat))
        .collect::<Vec<_>>();
    let levels = hierarchy.get_partitions().len();

    Box::new(move |r: &mut Array1<f64>| {
        let mut x_ks: Vec<Array1<f64>> = hierarchy
            .get_matrices()
            .iter()
            .map(|p| Array::from_vec(vec![0.0; p.rows()]))
            .collect();
        let mut b_ks = x_ks.clone();
        let p_ks = hierarchy.get_partitions();
        let mat_ks = hierarchy.get_matrices();
        b_ks[0] = r.clone();

        for level in 0..levels {
            for _ in 0..smoothing_steps {
                let mut r_k = &b_ks[level] - &(&mat_ks[level] * &x_ks[level]);
                smoothers[level](&mut r_k);
                x_ks[level] += &r_k;
            }
            let p_t = p_ks[level].transpose_view().to_owned();
            let r_k = &b_ks[level] - &(&mat_ks[level] * &x_ks[level]);
            b_ks[level + 1] = &p_t * &r_k;
        }

        x_ks[levels] = mat_coarse.solvec(&b_ks[levels]).unwrap();

        for level in (0..levels).rev() {
            let interpolated_x = hierarchy.get_partition(level) * &x_ks[level + 1];
            x_ks[level] += &interpolated_x;
            for _ in 0..smoothing_steps {
                let mut r_k = &b_ks[level] - &(hierarchy.get_matrix(level) * &x_ks[level]);
                smoothers[level](&mut r_k);
                x_ks[level] += &r_k;
            }
        }
        *r = x_ks.remove(0);
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        partitioner::modularity_matching,
        preconditioner::{l1, multilevelgs, multilevell1, sgs},
        solver::stationary,
    };
    use ndarray::{Array, Array1};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use test_generator::test_resources;

    fn test_symmetry<F>(preconditioner: &F, dim: usize)
    where
        F: Fn(&mut Array1<f64>),
    {
        for _ in 0..50 {
            let u = ndarray::Array::random(dim, Uniform::new(0., 2.));
            let v = ndarray::Array::random(dim, Uniform::new(0., 2.));
            let mut preconditioned_v = v.clone();
            let mut preconditioned_u = u.clone();
            preconditioner(&mut preconditioned_v);
            preconditioner(&mut preconditioned_u);

            let left: f64 = u.t().dot(&preconditioned_v);
            let right: f64 = v.t().dot(&preconditioned_u);
            assert!(
                (left - right).abs() < 10e-6,
                "Left: {}, Right: {}",
                left,
                right
            );
        }
    }

    fn test_positive_definiteness<F>(preconditioner: &F, dim: usize)
    where
        F: Fn(&mut Array1<f64>),
    {
        for _ in 0..50 {
            let u = ndarray::Array::random(dim, Uniform::new(0., 2.));
            let v = ndarray::Array::random(dim, Uniform::new(0., 2.));
            let mut preconditioned_v = v.clone();
            let mut preconditioned_u = u.clone();
            preconditioner(&mut preconditioned_v);
            preconditioner(&mut preconditioned_u);

            let pos: f64 = u.t().dot(&preconditioned_u);
            assert!(pos > 0.0);
            let pos: f64 = v.t().dot(&preconditioned_v);
            assert!(pos > 0.0);
        }
    }

    fn multilevel_loader(mat_path: &str) -> (super::Hierarchy, usize) {
        let mat = sprs::io::read_matrix_market::<f64, usize, _>(mat_path)
            .unwrap()
            .to_csr::<usize>();
        let dim = mat.rows();
        let ones = Array::from_vec(vec![1.0; mat.rows()]);
        let zeros = Array::from_vec(vec![0.0; mat.rows()]);
        let (near_null, _) = stationary(&mat, &zeros, &ones, 5, 10.0_f64.powi(-6), &l1(&mat));

        (modularity_matching(mat.clone(), &near_null, 2.0), dim)
    }

    #[test_resources("test_matrices/*")]
    fn test_symmetry_l1(mat_path: &str) {
        let mat = sprs::io::read_matrix_market::<f64, usize, _>(mat_path)
            .unwrap()
            .to_csr::<usize>();
        let dim = mat.rows();
        let preconditioner = l1(&mat);
        test_symmetry(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_positive_definiteness_l1(mat_path: &str) {
        let mat = sprs::io::read_matrix_market::<f64, usize, _>(mat_path)
            .unwrap()
            .to_csr::<usize>();
        let dim = mat.rows();
        let preconditioner = l1(&mat);
        test_positive_definiteness(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_symmetry_sgs(mat_path: &str) {
        let mat = sprs::io::read_matrix_market::<f64, usize, _>(mat_path)
            .unwrap()
            .to_csr::<usize>();
        let dim = mat.rows();
        let preconditioner = sgs(&mat);
        test_symmetry(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_positive_definiteness_sgs(mat_path: &str) {
        let mat = sprs::io::read_matrix_market::<f64, usize, _>(mat_path)
            .unwrap()
            .to_csr::<usize>();
        let dim = mat.rows();
        let preconditioner = sgs(&mat);
        test_positive_definiteness(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_symmetry_ml1(mat_path: &str) {
        let (hierarchy, dim) = multilevel_loader(mat_path);
        let preconditioner = multilevell1(hierarchy);
        test_symmetry(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_positive_definiteness_ml1(mat_path: &str) {
        let (hierarchy, dim) = multilevel_loader(mat_path);
        let preconditioner = multilevell1(hierarchy);
        test_positive_definiteness(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_symmetry_mgs(mat_path: &str) {
        let (hierarchy, dim) = multilevel_loader(mat_path);
        let preconditioner = multilevelgs(hierarchy);
        test_symmetry(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_positive_definiteness_mgs(mat_path: &str) {
        let (hierarchy, dim) = multilevel_loader(mat_path);
        let preconditioner = multilevelgs(hierarchy);
        test_positive_definiteness(&preconditioner, dim);
    }
}
