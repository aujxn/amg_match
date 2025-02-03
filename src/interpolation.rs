use crate::partitioner::Partition;
use nalgebra::base::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};

#[derive(Copy, Clone, Debug)]
pub enum InterpolationType {
    UnsmoothedAggregation,
    SmoothedAggregation((usize, f64)),
    Classical,
}

pub fn smoothed_aggregation(
    fine_mat: &CsrMatrix<f64>,
    partition: &Partition,
    near_null: &DVector<f64>,
    smoothing_steps: usize,
    jacobi_weight: f64,
) -> (DVector<f64>, CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>) {
    let n_fine = fine_mat.nrows();
    let n_coarse = partition.agg_to_node.len();

    let mut coarse_near_null: DVector<f64> = DVector::zeros(n_coarse);
    let mut diag_inv = fine_mat.diagonal_as_csr();
    diag_inv
        .values_mut()
        .iter_mut()
        .for_each(|val| *val = jacobi_weight * val.recip());

    for (coarse_i, agg) in partition.agg_to_node.iter().enumerate() {
        let r: f64 = agg.iter().map(|i| near_null[*i].powf(2.0)).sum();
        coarse_near_null[coarse_i] = r.sqrt();
    }

    let mut p = CooMatrix::new(n_fine, n_coarse);
    for (fine_idx, coarse_idx) in partition.node_to_agg.iter().cloned().enumerate() {
        p.push(
            fine_idx,
            coarse_idx,
            near_null[fine_idx] / coarse_near_null[coarse_idx],
        );
    }
    let mut p = CsrMatrix::from(&p);

    for _ in 0..smoothing_steps {
        p = &p - (&diag_inv * (fine_mat * &p));
    }

    let r = p.transpose();
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p, mat_coarse)
}

pub fn classical(
    _fine_mat: &CsrMatrix<f64>,
    _partition: &Partition,
    _near_null: &DVector<f64>,
) -> (DVector<f64>, CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>) {
    unimplemented!()
}
