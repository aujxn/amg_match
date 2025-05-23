use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::{f64, usize};

use ndarray_linalg::{InverseInto, Norm, QRInto};
use rand::Rng;

use crate::Matrix;
use crate::{partitioner::Partition, CooMatrix, CsrMatrix, Vector};

#[derive(Copy, Clone, Debug)]
pub enum InterpolationType {
    UnsmoothedAggregation,
    SmoothedAggregation((usize, f64)),
    Classical,
}

pub struct InterpolationInfo {
    pub min_entries_per_row: usize,
    pub max_entries_per_row: usize,
    pub avg_entries_per_row: f64,
    pub min_weight: f64,
    pub max_weight: f64,
    pub min_rowsum: f64,
    pub max_rowsum: f64,
}

impl InterpolationInfo {
    pub fn new(p: &CsrMatrix) -> Self {
        let mut min_entries_per_row = usize::MAX;
        let mut max_entries_per_row = 0;
        let mut max_weight = f64::MIN;
        let mut min_weight = f64::MAX;
        let mut max_rowsum = f64::MIN;
        let mut min_rowsum = f64::MAX;

        for (_, row) in p.outer_iterator().enumerate() {
            let row_entries = row.data().len();
            let rowsum: f64 = row.data().iter().sum();

            if row_entries > max_entries_per_row {
                max_entries_per_row = row_entries;
            }

            if row_entries < min_entries_per_row {
                min_entries_per_row = row_entries;
            }

            if rowsum > max_rowsum {
                max_rowsum = rowsum;
            }

            if rowsum < min_rowsum {
                min_rowsum = rowsum;
            }

            for (_, w) in row.iter() {
                if *w > max_weight {
                    max_weight = *w;
                }
                if *w < min_weight {
                    min_weight = *w;
                }
            }
        }
        let avg_entries_per_row = (p.data().len() as f64) / (p.rows() as f64);

        Self {
            min_entries_per_row,
            max_entries_per_row,
            avg_entries_per_row,
            min_weight,
            max_weight,
            min_rowsum,
            max_rowsum,
        }
    }

    pub fn display(interpolations: &Vec<Self>) {
        println!("Interpolation Matrix Information:");
        println!();

        println!(
            "{:>4}  {:>24}  {:>20}  {:>20}",
            "", "entries / row", "weight values", "rowsums"
        );

        println!(
            "{:>4}  {:>8}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
            "lev", "min", "max", "avg", "min", "max", "min", "max"
        );

        println!(
            "{:-<4}  {:-<8}  {:-<8}  {:-<8}  {:-<10}  {:-<10}  {:-<10}  {:-<10}",
            "", "", "", "", "", "", "", "",
        );

        // Now print each row of data
        for (level, info) in interpolations.iter().enumerate() {
            println!(
                "{:>4}  {:>8}  {:>8}  {:>8.1}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.2e}",
                level,
                info.min_entries_per_row,
                info.max_entries_per_row,
                info.avg_entries_per_row,
                info.min_weight,
                info.max_weight,
                info.min_rowsum,
                info.max_rowsum,
            );
        }
    }
}

pub fn smooth_interpolation(
    mat: &CsrMatrix,
    p: &mut CsrMatrix,
    smoothing_steps: usize,
    jacobi_weight: f64,
) {
    //let l1 = L1::new(mat);

    let mut diag_inv = CsrMatrix::eye(mat.rows());
    for (smoother_diag, mat_diag) in diag_inv
        .data_mut()
        .iter_mut()
        .zip(mat.diag_iter().map(|v| v.unwrap()))
    //.zip(l1.l1_inverse.iter())
    {
        *smoother_diag = jacobi_weight * mat_diag.recip();
        //*smoother_diag = *mat_diag;
    }

    for _ in 0..smoothing_steps {
        let ap = mat * &*p;
        let smoothed = &diag_inv * &ap;
        *p = &*p - &smoothed;
    }
    *p = p.to_csr();

    /*
    let mut p_pruned = CooMatrix::new((p.rows(), p.cols()));
    let max_w = 4;
    let delta = 0.2;
    for (i, row) in p.outer_iterator().enumerate() {
        if row.data().len() <= max_w {
            for (j, v) in row.iter() {
                p_pruned.add_triplet(i, j, *v);
            }
        }
    }
    */
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

    smooth_interpolation(fine_mat, &mut p, smoothing_steps, jacobi_weight);

    let r = p.transpose_view().to_csr();
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p.to_csr(), mat_coarse.to_csr())
}

pub fn block_jacobi(mat: &CsrMatrix, block_size: usize, p: &CsrMatrix) -> CsrMatrix {
    let ndofs = mat.rows();
    let n_blocks = ndofs / block_size;
    let mut d_inv = CooMatrix::new((ndofs, ndofs));
    //let mut d = CooMatrix::new((ndofs, ndofs));

    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;

        let mut block = Matrix::zeros((block_size, block_size));
        for i in 0..block_size {
            for j in 0..block_size {
                if let Some(v) = mat.get(start + i, start + j) {
                    block[[i, j]] = *v;
                }
            }
        }

        /*
        for i in 0..block_size {
            for j in 0..block_size {
                d.add_triplet(start + i, start + j, block[[i, j]]);
            }
        }
        */

        block = block.inv_into().unwrap();
        for i in 0..block_size {
            for j in 0..block_size {
                d_inv.add_triplet(start + i, start + j, block[[i, j]]);
            }
        }
    }

    /*
    let d = d.to_csr();
    let guess = Vector::random(mat.cols(), Uniform::new(-1., 1.));
    let max_iter = 15;
    let tol_basis = 1e-12;
    let tol_eig = 1e-12;
    */

    let d_inv = d_inv.to_csr();
    //let eigs = generalized_lanczos(&d_inv, mat, &guess, max_iter, tol_basis, tol_eig);

    //   let eigs = generalized_lanczos(mat, &d_inv, &guess, max_iter, tol_basis, tol_eig);
    //let eigs = generalized_lanczos(mat, &d, &guess, max_iter, tol_basis, tol_eig);
    //let eigs = generalized_lanczos(&d, mat, &guess, max_iter, tol_basis, tol_eig);
    //let norm = eigs[eigs.len() - 1];
    let mut d_inv_a = &d_inv * mat;
    //d_inv_a *= 4.0 / (3.0 * norm);
    //d_inv_a *= 3.0 / (4.0 * norm);
    //d_inv_a /= norm;
    d_inv_a *= 0.66;
    let smoothed = &d_inv_a * &*p;
    let new_p = &*p - &smoothed;
    let new_p = new_p.to_csr();
    new_p

    /* TODO need to rescale rows if trimming....
    let max_rowsize = 99999999;
    let mut trimmed = CooMatrix::new((new_p.rows(), new_p.cols()));
    for (i, row) in new_p.outer_iterator().enumerate() {
        let mut cutoff = 0.0;
        if row.data().len() > max_rowsize {
            let mut weights = Vec::from(row.data());
            weights.par_sort_by(|w1, w2| w2.abs().partial_cmp(&w1.abs()).unwrap());
            cutoff = weights[max_rowsize].abs();
        }

        for (j, w) in row.iter() {
            if w.abs() > cutoff {
                trimmed.add_triplet(i, j, *w);
            }
        }
    }

    trimmed.to_csr()
    */
}

pub fn smoothed_aggregation2(
    fine_mat: &CsrMatrix,
    partition: &Partition,
    block_size: usize,
    near_null: &Matrix,
) -> (Matrix, CsrMatrix, CsrMatrix, CsrMatrix) {
    let n_fine = fine_mat.rows();
    let n_coarse = partition.agg_to_node.len();
    let k = near_null.ncols();
    let mut coarse_near_null = Matrix::zeros((partition.agg_to_node.len() * k, k));

    /*
    let mut diag_inv = CsrMatrix::eye(n_fine);
    for (smoother_diag, mat_diag) in diag_inv
        .data_mut()
        .iter_mut()
        .zip(fine_mat.diag_iter().map(|v| v.unwrap()))
    {
        *smoother_diag = 0.66 * mat_diag.recip();
    }
    */

    let mut p = CooMatrix::new((n_fine, n_coarse * k));
    for (coarse_idx, nodes) in partition.agg_to_node.iter().enumerate() {
        let local_rows = nodes.len();
        assert!(local_rows >= k);
        /*
        if local_rows < k {
            local_rows = k;
        }
        */
        let mut local = Matrix::zeros((local_rows, k));
        for (local_j, j) in nodes.iter().copied().enumerate() {
            for (dest, src) in local.row_mut(local_j).into_iter().zip(near_null.row(j)) {
                *dest = *src;
            }
        }

        //println!("\n{:#.1}", local);
        let (q, r) = local.qr_into().unwrap();
        assert_eq!(q.ncols(), k);
        //println!("-----\n{:#.1}", q);
        //println!("-----\n{:#.1}", r);
        /*
        if q.ncols() < k {
            let extension = Matrix::from_elem((q.nrows(), k - q.ncols()), 1.0);
            q = concatenate(Axis(1), &[q.view(), extension.view()]).unwrap();
        }
        */

        for (i, r_row) in r.rows().into_iter().enumerate() {
            for (dest, src) in coarse_near_null
                .row_mut(coarse_idx * k + i)
                .iter_mut()
                .zip(r_row)
            {
                *dest = *src;
            }
        }

        for (local_i, fine_i) in nodes.iter().copied().enumerate() {
            let col_start = k * coarse_idx;
            for (offset_j, src) in q.row(local_i).iter().enumerate() {
                p.add_triplet(fine_i, col_start + offset_j, *src);
            }
        }
    }
    let mut p = p.to_csr();

    //let presmooth_info = InterpolationInfo::new(&p);
    //InterpolationInfo::display(&vec![presmooth_info]);

    /*
    let dense = p.to_dense();
    println!("{:#.1}", dense);
    */
    /*
    let ap = fine_mat * &p;
    let smoothed = &diag_inv * &ap;
    p = &p - &smoothed;
    */
    //if block_size == 6 {
    if block_size == 1 {
        smooth_interpolation(fine_mat, &mut p, 1, 0.66);
    } else {
        p = block_jacobi(fine_mat, block_size, &p);
    };
    //}
    //
    //let postsmooth_info = InterpolationInfo::new(&p);
    //InterpolationInfo::display(&vec![postsmooth_info]);

    /*
    let dense = p.to_dense();
    println!("{:#.1}", dense);
    */

    let r = p.transpose_view().to_csr();
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p.to_csr(), mat_coarse.to_csr())
}

#[derive(Debug, Clone, Copy)]
struct Strength(f64, usize);

impl Ord for Strength {
    fn cmp(&self, other: &Self) -> Ordering {
        if !self.0.is_finite() {
            panic!()
        }
        if !other.0.is_finite() {
            panic!()
        }

        if other.0 == self.0 {
            other.1.cmp(&self.1)
        } else if other.0 < self.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

impl PartialOrd for Strength {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Strength {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl Eq for Strength {}

pub fn classical(
    fine_mat: &CsrMatrix,
    near_null: &Vector,
) -> (Vector, CsrMatrix, CsrMatrix, CsrMatrix) {
    let mut current_mat: CsrMatrix = fine_mat.clone();
    let mut current_p = CsrMatrix::eye(fine_mat.rows());
    let mut current_near_null = near_null.clone();
    let mut prev_size = current_mat.rows();
    let mut starting_coarse_dofs: Option<BTreeSet<usize>> = None;
    let target_cf = 8.0;
    let target_size = (fine_mat.rows() as f64 / target_cf).ceil() as usize;
    loop {
        let (coarse_near_null, _r, p, ac, _, _) = classical_step(
            &current_mat,
            &current_near_null,
            &starting_coarse_dofs,
            target_size,
        );

        /*
        let (coarse_near_null, _r, p, ac) =
            pmis(&current_mat, &current_near_null, 0.5, target_size);
        */

        current_mat = ac.clone();
        current_p = &current_p * &p;
        current_p = current_p.to_csr();
        current_near_null = coarse_near_null.clone();
        let cf = current_p.rows() as f64 / current_p.cols() as f64;
        trace!("coarse size: {}, current cf: {:.2}", current_p.cols(), cf);

        if cf > target_cf {
            let mut p = current_p.clone();
            let test = &p * &current_near_null;
            /*
                    for (truth, interp) in near_null.iter().zip(test.iter()) {
                        println!("{:.3e} {:.3e}", truth, interp);
                    }
            */
            let err = near_null - test;
            info!("Final reconstruction error: {:.2e}", err.norm());
            let smooth_interp = true;
            if smooth_interp {
                smooth_interpolation(fine_mat, &mut p, 1, 0.66);
                let test = &p * &current_near_null;
                let err = near_null - test;
                info!(
                    "Final reconstruction error after smoothing: {:.2e}",
                    err.norm()
                );
            }

            let r = p.transpose_view().to_csr();
            let ac = &r * &(fine_mat * &p);
            return (current_near_null, r, p, ac.to_csr());
        }

        if prev_size == current_mat.rows() {
            warn!("Couldn't coarsen to desired size");
            return (
                coarse_near_null,
                current_p.transpose_view().to_csr(),
                current_p.to_csr(),
                ac,
            );
        }

        /*
        for row in current_p.outer_iterator() {
            let nci_count = row.nnz();
            if nci_count >= 40 {
                if let Some(starting) = starting_coarse_dofs.as_mut() {
                    starting.extend(row.indices().iter());
                }
            }
        }
        */

        prev_size = current_mat.rows();
    }
}

pub fn classical_step(
    fine_mat: &CsrMatrix,
    near_null: &Vector,
    starting_coarse_dofs: &Option<BTreeSet<usize>>,
    target_size: usize,
) -> (
    Vector,
    CsrMatrix,
    CsrMatrix,
    CsrMatrix,
    Vec<BTreeSet<usize>>,
    BTreeSet<usize>,
) {
    let threshold = 0.5;
    let ndofs_fine = fine_mat.rows();
    let mut fine_dofs = BTreeSet::new();
    let mut all_coarse_dofs = vec![];
    let mut all_coarse_dofs_set = BTreeSet::new();
    let mut remaining_dofs: BTreeSet<usize> = (0..ndofs_fine).collect();
    if let Some(starting_coarse_dofs) = starting_coarse_dofs {
        remaining_dofs = remaining_dofs
            .difference(&all_coarse_dofs[0])
            .copied()
            .collect();
        all_coarse_dofs.push(starting_coarse_dofs.clone())
    }

    // TODO filter by threshold to max?
    let mut strengths: Vec<BTreeSet<Strength>> = fine_mat
        .outer_iterator()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .filter(|(j, _)| *j != i)
                .map(|(j, w)| Strength(-(w * near_null[j]) / near_null[i], j))
                .filter(|strength_ij| strength_ij.0.is_finite() && strength_ij.0 > 0.0)
                .collect::<BTreeSet<Strength>>()
        })
        .collect();

    let mut deltas = vec![None; ndofs_fine];

    let mut rng = rand::rng();

    // find max strength for each dof and associated potential coarse dof
    for (i, strong_neighbors) in strengths.iter_mut().enumerate() {
        let delta_i = strong_neighbors.iter().copied().next();
        if let Some(max_strength) = delta_i {
            strong_neighbors.retain(|strength| strength.0 > threshold * max_strength.0);
        }

        assert!(delta_i.is_some());
        if let Some(mut strength) = delta_i {
            strength.0 += rng.random_range(0.0..1e-3);
            deltas[i] = Some(strength);
        } else {
            deltas[i] = delta_i;
        }
    }

    let mut unassigned = remaining_dofs.clone();
    while !remaining_dofs.is_empty() {
        let mut coarse_dofs = BTreeSet::new();
        let mut new_fine_dofs = BTreeSet::new();

        // find all locally maximum dofs and add them to the new fine set. since they are
        // locally maximum, any of their neighbors which are remaining are safe to add to
        // the new coarse set. in particular, we can add the neighbor which caused this vertex to
        // be locally maximal.
        for (i, row) in fine_mat.outer_iterator().enumerate() {
            if remaining_dofs.contains(&i) {
                if let Some(Strength(delta_i, neighbor)) = deltas[i] {
                    let mut fine_candidate = true;
                    for (j, _) in row
                        .iter()
                        .filter(|(j, _)| i != *j && remaining_dofs.contains(j))
                    {
                        if let Some(Strength(delta_j, _)) = deltas[j] {
                            if delta_j >= delta_i {
                                fine_candidate = false;
                                break;
                            }
                        }
                    }
                    if fine_candidate {
                        new_fine_dofs.insert(i);
                        unassigned.remove(&neighbor);
                        if all_coarse_dofs_set.insert(neighbor) {
                            coarse_dofs.insert(neighbor);
                        }

                        let check = unassigned.remove(&i);
                        assert!(check);
                    }
                }
            }
            if unassigned.len() + all_coarse_dofs_set.len() < target_size {
                coarse_dofs.extend(&unassigned);
                all_coarse_dofs_set.extend(&unassigned);
                remaining_dofs = BTreeSet::new();
                break;
            }
        }

        remaining_dofs = remaining_dofs.difference(&coarse_dofs).copied().collect();
        remaining_dofs = remaining_dofs.difference(&new_fine_dofs).copied().collect();

        for i in remaining_dofs.iter() {
            for strength in strengths[*i].iter() {
                if coarse_dofs.contains(&strength.1) {
                    new_fine_dofs.insert(*i);
                    unassigned.remove(i);
                    break;
                }
            }
            if unassigned.len() + all_coarse_dofs_set.len() < target_size {
                break;
            }
        }
        remaining_dofs = remaining_dofs.difference(&new_fine_dofs).copied().collect();

        all_coarse_dofs.push(coarse_dofs);
        fine_dofs.extend(new_fine_dofs);
    }

    let coarse_dofs = all_coarse_dofs_set;
    fine_dofs = fine_dofs.difference(&coarse_dofs).copied().collect();

    #[cfg(debug_assertions)]
    {
        let all_dofs_test: BTreeSet<usize> = fine_dofs.union(&coarse_dofs).copied().collect();
        assert_eq!(all_dofs_test.len(), fine_mat.rows());
    }

    let total_coarse: usize = all_coarse_dofs
        .iter()
        .map(|coarse_set| coarse_set.len())
        .sum();

    info!(
        "ndofs_fine: {}, fine size: {}, coarse total: {}, coarse sizes: {}",
        ndofs_fine,
        fine_dofs.len(),
        total_coarse,
        all_coarse_dofs
            .iter()
            .map(|coarse_set| format!("{} ", coarse_set.len()))
            .collect::<String>()
    );

    let (p, coarse_near_null) =
        classical_coarsen(fine_mat, &fine_dofs, &all_coarse_dofs, near_null);

    let r = p.transpose_view().to_csr();
    let mat_coarse = &r * &(fine_mat * &p);
    (
        coarse_near_null,
        r,
        p.to_csr(),
        mat_coarse.to_csr(),
        all_coarse_dofs,
        fine_dofs,
    )
}

fn classical_coarsen(
    mat: &CsrMatrix,
    fine_vertices: &BTreeSet<usize>,
    coarse_vertices: &Vec<BTreeSet<usize>>,
    near_null: &Vector,
) -> (CsrMatrix, Vector) {
    let nci_max = 3;
    let threshold = 0.5;

    let neighbors: Vec<BTreeSet<usize>> = mat
        .outer_iterator()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                //.filter(|(j, _v)| *j != i)
                .filter(|(j, v)| *j != i && *v * near_null[*j] / near_null[i] < 0.0)
                //.filter(|(j, v)| *j != i && **v < 0.0)
                .map(|(j, _)| j)
                .collect()
        })
        .collect();

    let all_coarse: BTreeSet<usize> =
        coarse_vertices
            .iter()
            .fold(BTreeSet::new(), |mut acc, coarse_set| {
                acc.extend(coarse_set);
                acc
            });

    let mut p = CooMatrix::new((mat.cols(), all_coarse.len()));

    let mut coarse_near_null = Vector::zeros(all_coarse.len());
    let mut ic_to_coarse = HashMap::new();
    for (coarse_idx, fine_idx) in all_coarse.iter().enumerate() {
        p.add_triplet(*fine_idx, coarse_idx, 1.0);
        coarse_near_null[coarse_idx] = near_null[*fine_idx];
        ic_to_coarse.insert(fine_idx, coarse_idx);
    }

    let mut strong = vec![BTreeSet::new(); fine_vertices.len()];
    for (i, fine_idx) in fine_vertices.iter().enumerate() {
        for (j, w) in mat.outer_view(*fine_idx).unwrap().iter() {
            if *fine_idx != j {
                let strength = -(w * near_null[j]) / near_null[*fine_idx];
                if strength.is_finite() && strength > 0.0 {
                    if *fine_idx != j {
                        strong[i].insert(j);
                    }
                }
            }
        }
    }

    let mut nc = vec![BTreeSet::new(); fine_vertices.len()];
    for i in 0..fine_vertices.len() {
        for coarse_set in coarse_vertices.iter() {
            if nc[i].len() >= nci_max {
                break;
            }
            for candidate in coarse_set.intersection(&strong[i]) {
                if nc[i].len() >= nci_max {
                    break;
                }
                nc[i].insert(*candidate);
            }
        }
    }

    #[cfg(debug_assertions)]
    for nci in nc.iter() {
        assert!(!nci.is_empty());
    }

    let strong_prime: Vec<BTreeSet<usize>> = strong
        .iter()
        .zip(nc.iter())
        .map(|(strong_neighbors, nci)| {
            strong_neighbors
                .difference(nci)
                .filter(|strong_neighbor| !neighbors[**strong_neighbor].is_disjoint(nci))
                .copied()
                .collect()
        })
        .collect();

    let deltas: Vec<f64> = fine_vertices
        .iter()
        .enumerate()
        .map(|(i, fine_i)| {
            nc[i]
                .iter()
                .chain(strong_prime[i].iter())
                .copied()
                .map(|fine_j| {
                    -(*mat.get(*fine_i, fine_j).unwrap() * near_null[fine_j]) / near_null[*fine_i]
                })
                .sum()
        })
        .collect();

    #[cfg(debug_assertions)]
    for delta_i in deltas.iter() {
        assert!(*delta_i > 0.0);
    }

    for (i, fine_i) in fine_vertices.iter().copied().enumerate() {
        for ic in nc[i].iter().copied() {
            let a_i_ic: f64 = *mat.get(fine_i, ic).unwrap();
            let mut weight: f64 = 0.0;

            for j in strong_prime[i].iter().copied() {
                let mut delta_jic = 0.0;
                if neighbors[j].contains(&ic) {
                    assert!(nc[i].intersection(&neighbors[j]).next().is_some());
                    for k in nc[i].intersection(&neighbors[j]).copied() {
                        if neighbors[j].contains(&k) {
                            delta_jic += mat.get(j, k).unwrap() * near_null[k];
                        }
                    }
                    //assert!(delta_jic < 0.0);
                    delta_jic = mat.get(j, ic).unwrap() / delta_jic;
                    assert!(delta_jic >= 0.0);
                    println!("{:.2e}", delta_jic);
                }

                weight += mat.get(fine_i, j).unwrap() * near_null[j] * delta_jic;
            }

            weight += a_i_ic;
            weight /= -deltas[i];

            let ic = *ic_to_coarse.get(&ic).unwrap();
            //println!("p_i,j: {} {} {:.2e}", fine_i, ic, weight);
            p.add_triplet(fine_i, ic, weight);
        }
    }

    let mut p: CsrMatrix = p.to_csr();
    let normalize_cols = false;
    if normalize_cols {
        let mut r = p.transpose_view().to_csr();

        for (coarse_i, mut row) in r.outer_iterator_mut().enumerate() {
            let scale: f64 = row.iter().map(|(_, v)| v.powf(2.0)).sum::<f64>().sqrt();
            row.iter_mut().for_each(|(_, v)| *v /= scale);
            coarse_near_null[coarse_i] *= scale;
        }
        p = r.transpose_into().to_csr();
    }
    (p, coarse_near_null)
}

/// Standard choice for alpha between 0.25 and 0.5:
/// (J. W. Ruge and K. St¨uben, Algebraic multigrid (AMG), in :
///     S. F. McCormick, ed., Multigrid Methods, vol. 3 of Frontiers in Applied Mathematics (SIAM, Philadelphia, 1987) 73–130.)
fn pmis(
    fine_mat: &CsrMatrix,
    near_null: &Vector,
    alpha: f64,
    target_size: usize,
) -> (Vector, CsrMatrix, CsrMatrix, CsrMatrix) {
    trace!("Starting pmis");
    let ndofs_fine = fine_mat.rows();
    let mut fine_dofs = BTreeSet::new();
    let mut all_coarse_dofs = vec![];
    let mut remaining_dofs: BTreeSet<usize> = (0..ndofs_fine).collect();

    let mut neighborhoods: Vec<BTreeSet<usize>> = fine_mat
        .outer_iterator()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .filter(|(j, w)| *j != i && **w * near_null[*j] / near_null[i] < 0.0)
                .map(|(j, _)| j)
                .collect()
        })
        .collect();

    let mut influenced_by: Vec<BTreeSet<Strength>> = vec![BTreeSet::new(); ndofs_fine];
    let mut influences: Vec<BTreeSet<Strength>> = vec![BTreeSet::new(); ndofs_fine];

    // populate non-symmetric strength relationship with bi-directional lookup
    for (i, row) in fine_mat.outer_iterator().enumerate() {
        for (strength_ij, j) in row
            .iter()
            .filter(|(j, _)| *j != i)
            .map(|(j, w)| (-(w * near_null[j]) / near_null[i], j))
            .filter(|strength_ij| strength_ij.0 > 0.0)
        {
            influenced_by[i].insert(Strength(strength_ij, j));
            influences[j].insert(Strength(strength_ij, i));
        }
    }

    let mut rng = rand::rng();
    let influence_total: Vec<f64> = influences
        .iter()
        .map(|strengths| {
            if let Some(max) = strengths.first() {
                strengths
                    .iter()
                    .map(|strength| strength.0 / max.0)
                    .sum::<f64>()
                    + rng.random_range(0.0..1e-5)
            } else {
                0.0
            }
        })
        .collect();

    let update_neighborhoods = |neighborhoods: &mut Vec<BTreeSet<usize>>,
                                to_remove: &BTreeSet<usize>| {
        for i in to_remove.iter().copied() {
            neighborhoods.push(BTreeSet::new());
            let neighbors = neighborhoods.swap_remove(i);
            for j in neighbors {
                neighborhoods[j].remove(&i);
            }
        }
    };

    // any vertices that don't strongly influence anyone else must be F-vertices
    // .... this is sketchy, how to interpolate them?
    /*
    for (i, influenced_by_i) in influences.iter().enumerate() {
        if influenced_by_i.is_empty() {
            fine_dofs.insert(i);
            remaining_dofs.remove(&i);
        }
    }

    if !fine_dofs.is_empty() {
        warn!(
            "{} dofs don't strongly influence any other dofs... making them F dofs",
            fine_dofs.len()
        );
    }
    // remove initial F-vertices from graph
    update_neighborhoods(&mut neighborhoods, &fine_dofs);
    */

    //let mut iter = 0;
    // choose F and C points until graph is covered
    while !remaining_dofs.is_empty() {
        /*
        iter += 1;
        trace!(
            "iter {}: Starting C/F selection by Luby's with {} remaining dofs.",
            iter,
            remaining_dofs.len(),
        );
        */
        let mut new_coarse_dofs = BTreeSet::new();
        let mut new_fine_dofs = BTreeSet::new();
        let mut to_remove = BTreeSet::new();

        // select an independent set of new coarse dofs
        for i in remaining_dofs.iter().copied() {
            let score_i = influence_total[i];
            let maximal = neighborhoods[i]
                .iter()
                .copied()
                .map(|j| influence_total[j])
                .all(|score_j| score_i > score_j);
            if maximal {
                new_coarse_dofs.insert(i);
                to_remove.insert(i);
            }
        }

        // find all fine dofs strongly influenced by selected coarse dofs
        for i in new_coarse_dofs.iter().copied() {
            for strength in influences[i].iter() {
                new_fine_dofs.insert(strength.1);
                to_remove.insert(strength.1);
            }
        }

        if new_fine_dofs.is_empty() && new_coarse_dofs.is_empty() {
            for i in remaining_dofs.iter().copied() {
                print!("{:.2e} ", influence_total[i]);
            }
            println!();
            panic!();
        }

        new_fine_dofs = new_fine_dofs
            .difference(&new_coarse_dofs)
            .copied()
            .collect();
        new_fine_dofs = new_fine_dofs
            .intersection(&remaining_dofs)
            .copied()
            .collect();
        to_remove = to_remove.intersection(&remaining_dofs).copied().collect();

        /*
                trace!(
                    "{} coarse dofs and {} fine dofs selected ({} total)",
                    new_coarse_dofs.len(),
                    new_fine_dofs.len(),
                    to_remove.len()
                );
        */
        // update neighborhoods and remaining dofs
        update_neighborhoods(&mut neighborhoods, &to_remove);
        remaining_dofs = remaining_dofs.difference(&to_remove).copied().collect();
        all_coarse_dofs.push(new_coarse_dofs);
        fine_dofs.extend(new_fine_dofs);
    }

    let (p, coarse_near_null) =
        classical_coarsen(fine_mat, &fine_dofs, &all_coarse_dofs, near_null);

    let r = p.transpose_view().to_csr();
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p.to_csr(), mat_coarse.to_csr())
}
