use crate::{
    partitioner::Hierarchy,
    solver::{lsolve, usolve},
};
use nalgebra::base::DVector;
use nalgebra_sparse::CsrMatrix;

pub fn l1(mat: &CsrMatrix<f64>) -> Box<dyn Fn(&mut DVector<f64>)> {
    let l1_inverse: Vec<f64> = mat
        .row_iter()
        .map(|row_vec| {
            let row_sum_abs: f64 = row_vec.values().iter().map(|val| val.abs()).sum();
            1.0 / row_sum_abs
        })
        .collect();
    let l1_inverse: DVector<f64> = DVector::from(l1_inverse);
    Box::new(move |r: &mut DVector<f64>| {
        r.component_mul_assign(&l1_inverse);
    })
}

pub fn fgs(mat: &CsrMatrix<f64>) -> Box<dyn Fn(&mut DVector<f64>)> {
    let lower_triangle = mat.lower_triangle();
    Box::new(move |r: &mut DVector<f64>| {
        lsolve(&lower_triangle, r);
    })
}

pub fn bgs(mat: &CsrMatrix<f64>) -> Box<dyn Fn(&mut DVector<f64>)> {
    let upper_triangle = mat.upper_triangle();
    Box::new(move |r: &mut DVector<f64>| {
        usolve(&upper_triangle, r);
    })
}

pub fn sgs<'a>(mat: &'a CsrMatrix<f64>) -> Box<dyn 'a + Fn(&mut DVector<f64>)> {
    let (_, _, diag) = mat.diagonal_as_csr().disassemble();
    let diag = DVector::from(diag);

    let lower_triangle = mat.lower_triangle();
    let upper_triangle = mat.upper_triangle();

    Box::new(move |r: &mut DVector<f64>| {
        lsolve(&lower_triangle, r);
        r.component_mul_assign(&diag);
        usolve(&upper_triangle, r);
    })
}

pub fn multilevelgs(hierarchy: Hierarchy) -> Box<dyn Fn(&mut DVector<f64>)> {
    let mat_coarse = nalgebra::DMatrix::from(hierarchy.get_matrices().last().unwrap());
    let decomp = mat_coarse.lu();
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

    Box::new(move |r: &mut DVector<f64>| {
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
            let p_t = partition_mat.transpose();
            let r_k = &b_ks[level] - &(mat * &x_ks[level]);
            b_ks.push(&p_t * &r_k);
        }

        let x_c = decomp.solve(&b_ks[levels]).unwrap();
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

pub fn multilevell1(hierarchy: Hierarchy) -> Box<dyn Fn(&mut DVector<f64>)> {
    let mat_coarse = nalgebra::DMatrix::from(hierarchy.get_matrices().last().unwrap());
    trace!("decomposing coarse problem");
    //let decomp = mat_coarse.lu();
    let decomp = nalgebra_lapack::LU::new(mat_coarse);
    let smoothing_steps = 3;
    trace!("building multilevel smoothers");
    let smoothers = hierarchy
        .get_matrices()
        .iter()
        .map(|mat| l1(mat))
        .collect::<Vec<_>>();
    let levels = hierarchy.get_partitions().len();

    Box::new(move |r: &mut DVector<f64>| {
        let mut x_ks: Vec<DVector<f64>> = hierarchy
            .get_matrices()
            .iter()
            .map(|p| DVector::from(vec![0.0; p.nrows()]))
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
            let p_t = p_ks[level].transpose();
            let r_k = &b_ks[level] - &(&mat_ks[level] * &x_ks[level]);
            b_ks[level + 1] = &p_t * &r_k;
        }

        x_ks[levels] = decomp.solve(&b_ks[levels]).unwrap();

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
    use nalgebra::base::DVector;
    use nalgebra_sparse::csr::CsrMatrix;
    use rand::{distributions::Uniform, thread_rng};
    use test_generator::test_resources;

    fn test_symmetry<F>(preconditioner: &F, dim: usize)
    where
        F: Fn(&mut DVector<f64>),
    {
        let mut rng = thread_rng();
        let distribution = Uniform::new(-2.0_f64, 2.0_f64);
        for _ in 0..50 {
            let u: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
            let v: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
            let mut preconditioned_v = v.clone();
            let mut preconditioned_u = u.clone();
            preconditioner(&mut preconditioned_v);
            preconditioner(&mut preconditioned_u);

            let left: f64 = u.dot(&preconditioned_v);
            let right: f64 = v.dot(&preconditioned_u);
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
        F: Fn(&mut DVector<f64>),
    {
        let mut rng = thread_rng();
        let distribution = Uniform::new(-2.0_f64, 2.0_f64);
        for _ in 0..50 {
            let u: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
            let v: DVector<f64> = DVector::from_distribution(dim, &distribution, &mut rng);
            let mut preconditioned_v = v.clone();
            let mut preconditioned_u = u.clone();
            preconditioner(&mut preconditioned_v);
            preconditioner(&mut preconditioned_u);

            let pos: f64 = u.dot(&preconditioned_u);
            assert!(pos > 0.0);
            let pos: f64 = v.dot(&preconditioned_v);
            assert!(pos > 0.0);
        }
    }

    fn multilevel_loader(mat_path: &str) -> (super::Hierarchy, usize) {
        let mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(mat_path).unwrap(),
        );
        let dim = mat.nrows();
        let ones = DVector::from(vec![1.0; mat.nrows()]);
        let zeros = DVector::from(vec![0.0; mat.nrows()]);
        let (near_null, _) = stationary(&mat, &zeros, &ones, 5, 10.0_f64.powi(-6), &l1(&mat));

        (modularity_matching(mat.clone(), &near_null, 2.0), dim)
    }

    #[test_resources("test_matrices/*")]
    fn test_symmetry_l1(mat_path: &str) {
        let mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(mat_path).unwrap(),
        );
        let dim = mat.nrows();
        let preconditioner = l1(&mat);
        test_symmetry(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_positive_definiteness_l1(mat_path: &str) {
        let mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(mat_path).unwrap(),
        );
        let dim = mat.nrows();
        let preconditioner = l1(&mat);
        test_positive_definiteness(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_symmetry_sgs(mat_path: &str) {
        let mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(mat_path).unwrap(),
        );
        let dim = mat.nrows();
        let preconditioner = sgs(&mat);
        test_symmetry(&preconditioner, dim);
    }

    #[test_resources("test_matrices/*")]
    fn test_positive_definiteness_sgs(mat_path: &str) {
        let mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(mat_path).unwrap(),
        );
        let dim = mat.nrows();
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
