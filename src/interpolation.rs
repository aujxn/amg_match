use crate::{partitioner::Partition, CooMatrix, CsrMatrix, Vector};

#[derive(Copy, Clone, Debug)]
pub enum InterpolationType {
    UnsmoothedAggregation,
    SmoothedAggregation((usize, f64)),
    Classical,
}

pub fn smoothed_aggregation(
    fine_mat: &CsrMatrix,
    partition: &Partition,
    near_null: &Vector,
    smoothing_steps: usize,
    jacobi_weight: f64,
) -> (Vector, CsrMatrix, CsrMatrix, CsrMatrix) {
    let n_fine = fine_mat.rows();
    let n_coarse = partition.agg_to_node.len();

    let mut coarse_near_null: Vector = Vector::zeros(n_coarse);
    let mut diag_inv = CsrMatrix::eye(n_fine);
    for (smoother_diag, mat_diag) in diag_inv
        .data_mut()
        .iter_mut()
        .zip(fine_mat.diag_iter().map(|v| v.unwrap()))
    {
        *smoother_diag = -jacobi_weight * mat_diag.recip();
    }

    for (coarse_i, agg) in partition.agg_to_node.iter().enumerate() {
        let r: f64 = agg.iter().map(|i| near_null[*i].powf(2.0)).sum();
        coarse_near_null[coarse_i] = r.sqrt();
    }

    let mut p = CooMatrix::new((n_fine, n_coarse));
    for (fine_idx, coarse_idx) in partition.node_to_agg.iter().cloned().enumerate() {
        p.add_triplet(
            fine_idx,
            coarse_idx,
            near_null[fine_idx] / coarse_near_null[coarse_idx],
        );
    }
    let mut p = p.to_csr();

    for _ in 0..smoothing_steps {
        let ap = fine_mat * &p;
        let smoothed = &diag_inv * &ap;
        p = &p - &smoothed;
    }

    let r = p.transpose_view().to_csr();
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p, mat_coarse)
}

pub fn classical(
    _fine_mat: &CsrMatrix,
    _partition: &Partition,
    _near_null: &Vector,
) -> (Vector, CsrMatrix, CsrMatrix, CsrMatrix) {
    unimplemented!()
}
