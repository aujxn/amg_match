use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use ndarray_linalg::{assert, Norm};

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
        *smoother_diag = jacobi_weight * mat_diag.recip();
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
    (coarse_near_null, r, p.to_csr(), mat_coarse.to_csr())
}

#[derive(Debug)]
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
            Ordering::Equal
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
    let mut starting_coarse_dofs = None;
    let target_cf = 8.0;
    let target_size = (fine_mat.rows() as f64 / target_cf).ceil() as usize;
    loop {
        let (coarse_near_null, _r, p, ac) = classical_step(
            &current_mat,
            &current_near_null,
            &starting_coarse_dofs,
            target_size,
        );
        current_mat = ac.clone();
        current_p = &current_p * &p;
        current_p = current_p.to_csr();
        current_near_null = coarse_near_null.clone();
        let cf = current_p.rows() as f64 / current_p.cols() as f64;
        trace!("coarse size: {}, current cf: {:.2}", current_p.cols(), cf);

        if cf > target_cf {
            let mut r = current_p.transpose_view().to_csr();
            for (coarse_i, mut row) in r.outer_iterator_mut().enumerate() {
                let scale: f64 = row.iter().map(|(_, v)| v.powf(2.0)).sum::<f64>().sqrt();
                row.iter_mut().for_each(|(_, v)| *v /= scale);
                current_near_null[coarse_i] *= scale;
            }
            let p = r.transpose_view().to_csr();
            let ac = &r * &(fine_mat * &p);
            let test = &p * &current_near_null;
            let err = near_null - test;
            info!("Final reconstruction error: {:.2e}", err.norm());
            return (current_near_null, r, p, ac);
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

        for row in current_p.outer_iterator() {
            let nci_count = row.nnz();
            if nci_count >= 40 {
                if let Some(starting) = starting_coarse_dofs.as_mut() {
                    starting.extend(row.indices().iter());
                }
            }
        }

        prev_size = current_mat.rows();
    }
}

pub fn classical_step(
    fine_mat: &CsrMatrix,
    near_null: &Vector,
    starting_coarse_dofs: &Option<BTreeSet<usize>>,
    target_size: usize,
) -> (Vector, CsrMatrix, CsrMatrix, CsrMatrix) {
    let ndofs_fine = fine_mat.rows();
    let mut fine_dofs = BTreeSet::new();
    let mut all_coarse_dofs = vec![];
    let mut remaining_dofs: BTreeSet<usize> = (0..ndofs_fine).collect();
    if let Some(starting_coarse_dofs) = starting_coarse_dofs {
        remaining_dofs = remaining_dofs
            .difference(&all_coarse_dofs[0])
            .copied()
            .collect();
        all_coarse_dofs.push(starting_coarse_dofs.clone())
    }
    let thresh = 0.5;

    let strengths: Vec<BTreeSet<Strength>> = fine_mat
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

    while !remaining_dofs.is_empty() {
        let mut coarse_dofs = BTreeSet::new();
        let mut new_fine_dofs = BTreeSet::new();
        let mut deltas = vec![None; ndofs_fine];

        #[cfg(debug_assertions)]
        for coarse_set in all_coarse_dofs.iter() {
            assert!(fine_dofs.intersection(coarse_set).next().is_none());
        }

        // find max strength for each dof and associated potential coarse dof
        for (i, row) in fine_mat
            .outer_iterator()
            .enumerate()
            .filter(|(i, _)| remaining_dofs.contains(i))
        {
            let mut delta_i = None;
            for (j, w) in row
                .iter()
                .filter(|(j, _)| i != *j && remaining_dofs.contains(j))
            {
                let strength = -(w * near_null[j]) / near_null[i];
                if strength.is_finite() && strength > 0.0 {
                    if let Some((old_strength, _)) = delta_i {
                        if old_strength < strength {
                            delta_i = Some((strength, j));
                        }
                    } else {
                        delta_i = Some((strength, j));
                    }
                }
            }
            deltas[i] = delta_i;
        }

        // find all locally maximum dofs and add them to the new fine set. since they are
        // locally maximum, any of their neighbors which are remaining are safe to add to
        // the new coarse set. in particular, we can add the neighbor which caused this vertex to
        // be locally maximal.
        for (i, row) in fine_mat.outer_iterator().enumerate() {
            if remaining_dofs.contains(&i) {
                if let Some((delta_i, neighbor)) = deltas[i] {
                    let mut fine_candidate = true;
                    for (j, _) in row
                        .iter()
                        .filter(|(j, _)| i != *j && remaining_dofs.contains(j))
                    {
                        if let Some((delta_j, _)) = deltas[j] {
                            if delta_j >= delta_i {
                                fine_candidate = false;
                                break;
                            }
                        }
                    }
                    if fine_candidate {
                        new_fine_dofs.insert(i);
                        coarse_dofs.insert(neighbor);
                    }
                }
            }
            if remaining_dofs.len() - new_fine_dofs.len() < target_size {
                break;
            }
        }

        remaining_dofs = remaining_dofs.difference(&coarse_dofs).copied().collect();
        remaining_dofs = remaining_dofs.difference(&new_fine_dofs).copied().collect();
        let mut fine_counter = 0;
        for i in remaining_dofs.iter() {
            if let Some((_delta_i, j)) = deltas[*i] {
                if coarse_dofs.contains(&j) {
                    new_fine_dofs.insert(*i);
                    fine_counter += 1;
                }
            } else {
                if let Some(Strength(max_strength, _)) = strengths[*i].first() {
                    for strength in strengths[*i].iter() {
                        if strength.0 < thresh * max_strength {
                            break;
                        }
                        if all_coarse_dofs
                            .iter()
                            .any(|coarse_set| coarse_set.contains(&strength.1))
                        {
                            new_fine_dofs.insert(*i);
                            fine_counter += 1;
                            break;
                        }
                    }
                }
            }
            if remaining_dofs.len() - fine_counter < target_size {
                break;
            }
        }
        remaining_dofs = remaining_dofs.difference(&new_fine_dofs).copied().collect();

        if new_fine_dofs.is_empty() {
            all_coarse_dofs.push(remaining_dofs);
            remaining_dofs = BTreeSet::new();
        } else {
            all_coarse_dofs.push(coarse_dofs);
            fine_dofs.extend(new_fine_dofs);
        }
    }

    #[cfg(debug_assertions)]
    {
        let mut all_dofs_test = fine_dofs.clone();
        for coarse_set in all_coarse_dofs.iter() {
            assert!(fine_dofs.intersection(coarse_set).next().is_none());
            all_dofs_test.extend(coarse_set);
        }
        for i in 0..ndofs_fine {
            assert!(all_dofs_test.contains(&i));
        }
    }

    let total_coarse: usize = all_coarse_dofs
        .iter()
        .map(|coarse_set| coarse_set.len())
        .sum();
    /*
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
    */

    let (p, coarse_near_null) = classical_coarsen(fine_mat, fine_dofs, all_coarse_dofs, near_null);

    let r = p.transpose_view().to_csr();
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p.to_csr(), mat_coarse.to_csr())
}

fn classical_coarsen(
    mat: &CsrMatrix,
    fine_vertices: BTreeSet<usize>,
    coarse_vertices: Vec<BTreeSet<usize>>,
    near_null: &Vector,
) -> (CsrMatrix, Vector) {
    let nci_max = 2;

    let neighbors: Vec<BTreeSet<usize>> = mat
        .outer_iterator()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .filter(|(j, _v)| *j != i)
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
            let strength = -(w * near_null[j]) / near_null[*fine_idx];
            if strength.is_finite() && strength > 0.0 {
                if *fine_idx != j {
                    strong[i].insert(j);
                }
            }
        }
    }

    let mut nc = vec![BTreeSet::new(); fine_vertices.len()];
    for i in 0..fine_vertices.len() {
        for (k, coarse_set) in coarse_vertices.iter().enumerate() {
            if k == coarse_vertices.len() - 1 {
                break;
            }
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
                    -*mat.get(*fine_i, fine_j).unwrap() * near_null[fine_j] / near_null[*fine_i]
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
                assert!(nc[i].intersection(&neighbors[j]).next().is_some());
                for k in nc[i].intersection(&neighbors[j]).copied() {
                    delta_jic += mat.get(j, k).unwrap() * near_null[k];
                }
                //assert!(delta_jic < 0.0);
                delta_jic = mat.get(j, ic).unwrap_or(&0.0) / delta_jic;
                //assert!(delta_jic >= 0.0);

                weight += mat.get(fine_i, j).unwrap() * near_null[j] * delta_jic;
            }

            weight += a_i_ic;
            weight /= -deltas[i];

            p.add_triplet(fine_i, *ic_to_coarse.get(&ic).unwrap(), weight);
        }
    }

    let p: CsrMatrix = p.to_csr();
    let mut r = p.transpose_view().to_csr();

    for (coarse_i, mut row) in r.outer_iterator_mut().enumerate() {
        let scale: f64 = row.iter().map(|(_, v)| v.powf(2.0)).sum::<f64>().sqrt();
        row.iter_mut().for_each(|(_, v)| *v /= scale);
        coarse_near_null[coarse_i] *= scale;
    }
    let p = r.transpose_into().to_csr();
    //let test = &p * &coarse_near_null;
    //let err = near_null - test;
    //info!("Reconstruction error: {:.2e}", err.norm());
    (p, coarse_near_null)
}
