//! This module contains methods that partitions the matrix into hierarchies.
//! This could potentially be moved to a seperate crate, since I have copied
//! this code into other projects as well.

use core::fmt;
use indexmap::IndexSet;
use nalgebra::base::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use std::borrow::Borrow;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::parallel_ops::spmm;

//TODO bring back tests you deleted when preconditioner refactor happened
//     also, stop copying the fine mat into the hierarchy

/// Resulting object from running the modularity matching algorithm.
/// NOTE: Maybe don't store each matrix and just provide the P's.
#[derive(Clone)]
pub struct Hierarchy {
    mat: Rc<CsrMatrix<f64>>,
    partition_matrices: Vec<CsrMatrix<f64>>,
    interpolation_matrices: Vec<CsrMatrix<f64>>,
    pub matrices: Vec<Rc<CsrMatrix<f64>>>,
}

impl fmt::Debug for Hierarchy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sizes: Vec<usize> = self.partition_matrices.iter().map(|p| p.nrows()).collect();
        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.matrices.iter().map(|p| p.nrows()));
        let total_nnz_coarse = nnzs.iter().sum::<usize>() as f32;
        let complexity = total_nnz_coarse / (fine_nnz as f32);
        let coarsening_factors: Vec<f32> = sizes
            .iter()
            .zip(sizes.iter().skip(1))
            .map(|(a, b)| (*a as f32) / (*b as f32))
            .collect();

        f.debug_struct("Hierarchy")
            .field("levels", &self.levels())
            .field("sizes", &sizes)
            .field("coarsening_factors", &coarsening_factors)
            .field("nnz", &nnzs)
            .field("total_nnz_and_complexity", &(total_nnz_coarse, complexity))
            .finish()
    }
}

impl Hierarchy {
    pub fn new(mat: Rc<CsrMatrix<f64>>) -> Self {
        Self {
            mat,
            partition_matrices: Vec::new(),
            interpolation_matrices: Vec::new(),
            matrices: Vec::new(),
        }
    }

    pub fn from_hierarchy(
        mat: Rc<CsrMatrix<f64>>,
        partition_matrices: Vec<CsrMatrix<f64>>,
    ) -> Self {
        let interpolation_matrices: Vec<CsrMatrix<f64>> =
            partition_matrices.iter().map(|p| p.transpose()).collect();
        let mut matrices: Vec<Rc<CsrMatrix<f64>>> = Vec::new();

        for (p, p_t) in partition_matrices.iter().zip(interpolation_matrices.iter()) {
            if let Some(mat) = matrices.last() {
                let rc: Rc<CsrMatrix<f64>> = mat.clone();
                let prev: &CsrMatrix<f64> = rc.borrow();
                let coarse_mat = p_t * &(prev * p);
                matrices.push(Rc::new(coarse_mat));
            } else {
                let fine_mat: &CsrMatrix<f64> = mat.borrow();
                let coarse_mat = p_t * &(fine_mat * p);
                matrices.push(Rc::new(coarse_mat));
            }
        }

        Self {
            mat,
            partition_matrices,
            interpolation_matrices,
            matrices,
        }
    }

    /// Number of levels in the hierarchy.
    pub fn levels(&self) -> usize {
        self.matrices.len() + 1
    }

    /// Check if the hierarchy has any levels
    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    /// Adds a level to the hierarchy.
    pub fn push(&mut self, partition_mat: CsrMatrix<f64>, near_null: &DVector<f64>) {
        let fine_mat: &CsrMatrix<f64>;
        if self.matrices.is_empty() {
            let mut partition_mat = partition_mat.clone();
            fine_mat = self.mat.borrow();
            fix_bad_clusters(&mut partition_mat, near_null, fine_mat);
            partition_mat
                .triplet_iter_mut()
                .for_each(|(i, _, w)| *w *= near_null[i]);

            // test product of all Ps with coarse ones to get w
            let ones = DVector::from(vec![1.0; partition_mat.ncols()]);
            let maybe_w = &partition_mat * &ones;
            for (w_i, maybe_w_i) in near_null.iter().zip(maybe_w.iter()) {
                assert!((maybe_w_i - w_i).abs() < 1e-10);
            }
        } else {
            fine_mat = self.matrices.last().unwrap().borrow();
        }

        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &partition_mat);
        self.matrices.push(Rc::new(coarse_mat));
        self.partition_matrices.push(partition_mat);
        self.interpolation_matrices.push(p_transpose);
    }

    pub fn push_p_scaled_by_w(
        &mut self,
        mut partition_mat: CsrMatrix<f64>,
        near_null: &DVector<f64>,
    ) {
        let fine_mat: &CsrMatrix<f64>;
        if self.matrices.is_empty() {
            fine_mat = self.mat.borrow();
        } else {
            fine_mat = self.matrices.last().unwrap().borrow();
        }

        fix_bad_clusters(&mut partition_mat, near_null, fine_mat);
        partition_mat
            .triplet_iter_mut()
            .for_each(|(i, _, w)| *w *= near_null[i]);

        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &partition_mat);
        self.matrices.push(Rc::new(coarse_mat));
        self.partition_matrices.push(partition_mat);
        self.interpolation_matrices.push(p_transpose);
    }

    /// More advanced interpolation than piecewise constants.
    pub fn add_classical_interpolant(
        &mut self,
        partition_mat: CsrMatrix<f64>,
        near_null: &DVector<f64>,
    ) {
        let fine_mat: &CsrMatrix<f64>;

        if self.matrices.is_empty() {
            fine_mat = self.mat.borrow();
        } else {
            fine_mat = self.matrices.last().unwrap().borrow();
        }

        let n = fine_mat.nrows();
        let p_transpose = partition_mat.transpose();
        let mut strong_neighbors: Vec<IndexSet<usize>> = vec![IndexSet::new(); n];
        //let n_aggs = p_transpose.nrows();
        //let mut coarse_vertices_by_agg: Vec<IndexSet<usize>> = vec![IndexSet::new(); n_aggs];
        let mut delta_i = vec![0.0; n];
        let mut all_coarse = IndexSet::new();

        for agg in p_transpose.row_iter() {
            let mut coarse_vertices = IndexSet::new();
            let agg_vertices: IndexSet<usize> = agg.col_indices().iter().cloned().collect();
            let mut remaining: IndexSet<usize> = agg_vertices.clone();
            for i in agg_vertices.iter() {
                if near_null[*i] == 0.0 {
                    coarse_vertices.insert(*i);
                } else {
                    let mut neighbors: IndexSet<usize> = fine_mat
                        .row(*i)
                        .col_indices()
                        .iter()
                        .filter(|j| **j != *i)
                        .cloned()
                        .collect();
                    neighbors = neighbors.intersection(&agg_vertices).cloned().collect();
                    if neighbors
                        .iter()
                        .map(|j| fine_mat.index_entry(*i, *j).into_value() * near_null[*j])
                        .all(|prod| prod.abs() < 1e-12)
                    {
                        coarse_vertices.insert(*i);
                        for neighbor in neighbors.iter() {
                            remaining.swap_remove(neighbor);
                        }
                    }
                }
            }

            loop {
                if remaining.is_empty() {
                    break;
                }
                let new_coarse_vertex = remaining.pop().unwrap();
                coarse_vertices.insert(new_coarse_vertex);
                for neighbor in fine_mat.row(new_coarse_vertex).col_indices() {
                    remaining.swap_remove(neighbor);
                }
            }

            for i in agg_vertices.iter() {
                if coarse_vertices.contains(i) {
                    continue;
                }

                let neighborhood = fine_mat.row(*i);
                let mut negative_strengths = Vec::new();
                let mut total_strength = 0.0;
                for (a_ij, j) in neighborhood
                    .values()
                    .iter()
                    .zip(neighborhood.col_indices())
                    .filter(|(_, j)| **j != *i)
                {
                    if agg_vertices.contains(j) {
                        let strength = -a_ij * near_null[*j] / near_null[*i];
                        // todo if strength is massive then problems
                        if strength > 1e8 {
                            //warn!("Massive strength detected");
                        }
                        if strength > 0.0 {
                            strong_neighbors[*i].insert(*j);
                            total_strength += strength;
                        } else {
                            negative_strengths.push((j, strength));
                        }
                    }
                }
                if strong_neighbors[*i].is_empty() {
                    coarse_vertices.insert(*i);
                } else {
                    negative_strengths.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    loop {
                        if negative_strengths.is_empty() {
                            break;
                        }
                        let (j, strength) = negative_strengths.pop().unwrap();
                        total_strength += strength;
                        if total_strength < 0.0 {
                            break;
                        }
                        strong_neighbors[*i].insert(*j);
                    }
                    assert_eq!(delta_i[*i], 0.0);
                    for j in strong_neighbors[*i].iter() {
                        delta_i[*i] += fine_mat.index_entry(*i, *j).into_value() * near_null[*j];
                    }
                    delta_i[*i] /= -near_null[*i];
                }
            }
            for i in agg_vertices.iter() {
                if near_null[*i] == 0.0 {
                    continue;
                }
                if coarse_vertices.contains(i) {
                    continue;
                }
                if coarse_vertices.is_disjoint(&strong_neighbors[*i]) {
                    coarse_vertices.insert(*i);
                }
            }

            fn fix_si(
                agg_vertices: &IndexSet<usize>,
                coarse_vertices: &mut IndexSet<usize>,
                strong_neighbors: &Vec<IndexSet<usize>>,
                fine_mat: &CsrMatrix<f64>,
                near_null: &DVector<f64>,
            ) {
                for i in agg_vertices.iter() {
                    if coarse_vertices.contains(i) {
                        continue;
                    }

                    let si = &strong_neighbors[*i];
                    let mut aci: IndexSet<usize> =
                        coarse_vertices.intersection(si).cloned().collect();
                    let mut si_minus_aci: IndexSet<usize> = si.difference(&aci).cloned().collect();

                    loop {
                        let j = match si_minus_aci.iter().cloned().find(|j| {
                            let mut sum = 0.0;
                            for jc in aci.iter() {
                                sum += (fine_mat.index_entry(*j, *jc).into_value()
                                    * near_null[*jc])
                                    .abs();
                            }
                            sum == 0.0
                        }) {
                            Some(j) => j,
                            None => break,
                        };
                        coarse_vertices.insert(j);
                        aci.insert(j);
                        si_minus_aci.swap_remove(&j);
                    }
                }
            }
            fix_si(
                &agg_vertices,
                &mut coarse_vertices,
                &strong_neighbors,
                fine_mat,
                near_null,
            );

            for new_coarse in coarse_vertices {
                all_coarse.insert(new_coarse);
            }
        }

        let coarse_size = all_coarse.len();
        let mut interpolation = CooMatrix::new(n, coarse_size);
        for (coarse_idx, idx) in all_coarse.iter().enumerate() {
            interpolation.push(*idx, coarse_idx, 1.0);
        }
        let new_p = CsrMatrix::from(&interpolation);
        let new_pt = new_p.transpose();

        for i in 0..n {
            if all_coarse.contains(&i) || near_null[i] == 0.0 {
                continue;
            }
            let aci: IndexSet<usize> = strong_neighbors[i]
                .intersection(&all_coarse)
                .cloned()
                .collect();
            assert!(!aci.is_empty());

            for ic in aci.iter() {
                let mut value = fine_mat.index_entry(i, *ic).into_value();
                for j in strong_neighbors[i].difference(&aci) {
                    let sign = if near_null[*ic] > 0.0 {
                        1.0
                    } else if near_null[*ic] < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let numerator = fine_mat.index_entry(i, *j).into_value()
                        * near_null[*j]
                        * fine_mat.index_entry(*j, *ic).into_value().abs()
                        * sign;
                    let mut denom = 0.0;
                    for jc in aci.iter() {
                        denom +=
                            (fine_mat.index_entry(*j, *jc).into_value() * near_null[*jc]).abs();
                    }
                    assert!(denom != 0.0);
                    value += numerator / denom;
                }
                value /= -delta_i[i];
                assert!(!value.is_nan());
                let jc = all_coarse.get_index_of(ic).unwrap();
                interpolation.push(i, jc, value);
            }
        }

        let interpolation = CsrMatrix::from(&interpolation);
        let wc = &new_pt * near_null;
        let reconstruction = &interpolation * wc;

        let err = reconstruction - near_null;
        let err_norm = err.norm();
        let rel_err = err_norm / near_null.norm();
        if rel_err > 1e-4 {
            warn!(
                "Near-null reconstruction relative error norm: {:.2e}",
                rel_err
            );
        }

        let p_transpose = interpolation.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &interpolation);
        self.matrices.push(Rc::new(coarse_mat));
        //self.partition_matrices.push(new_p);
        //self.interpolation_matrices.push(new_pt);
        self.partition_matrices.push(interpolation);
        self.interpolation_matrices.push(p_transpose);
    }

    /// Get a single P matrix from the hierarchy.
    pub fn get_partition(&self, level: usize) -> &CsrMatrix<f64> {
        &self.partition_matrices[level]
    }

    /// Get a single P^T matrix from the hierarchy.
    pub fn get_interpolation(&self, level: usize) -> &CsrMatrix<f64> {
        &self.interpolation_matrices[level]
    }

    /// Get a reference to the matrices Vec.
    pub fn get_matrices(&self) -> &[Rc<CsrMatrix<f64>>] {
        &self.matrices
    }

    /// Get a reference to the P matrices Vec.
    pub fn get_partitions(&self) -> &Vec<CsrMatrix<f64>> {
        &self.partition_matrices
    }

    /// Get a reference to the P^T matrices Vec.
    pub fn get_interpolations(&self) -> &Vec<CsrMatrix<f64>> {
        &self.interpolation_matrices
    }

    pub fn get_mat(&self, level: usize) -> Rc<CsrMatrix<f64>> {
        if level == 0 {
            self.mat.clone()
        } else {
            self.matrices[level - 1].clone()
        }
    }
}

pub fn modularity_matching_add_level(
    near_null: &'_ DVector<f64>,
    coarsening_factor: f64,
    hierarchy: &mut Hierarchy,
) -> bool {
    let mat = hierarchy
        .get_matrices()
        .last()
        .map_or(hierarchy.get_mat(0), |m| m.clone());
    let (mut a_bar, mut row_sums, inverse_total) = build_weighted_matrix(&mat, &near_null);
    let mut modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
    let dim = mat.nrows();
    let mut partition_mat = CsrMatrix::identity(dim);
    let starting_vertex_count = dim as f64;

    loop {
        let vertex_count = modularity_mat.nrows();

        match find_pairs(&modularity_mat, 1) {
            None => {
                info!(
                    "Maximum modularity obtained. Levels: {}",
                    hierarchy.levels()
                );
                return false;
            }
            Some(pairs) => {
                let new_partition = build_partition_from_pairs(&pairs, vertex_count);
                let coarse_vertex_count = new_partition.ncols() as f64;
                partition_mat = &partition_mat * &new_partition;

                if starting_vertex_count / coarse_vertex_count > coarsening_factor {
                    //hierarchy.push_p_scaled_by_w(partition_mat, near_null);
                    hierarchy.add_classical_interpolant(partition_mat, near_null);
                    trace!(
                        "added level: {}. num vertices coarse: {} nnz: {}",
                        hierarchy.levels(),
                        hierarchy.get_partitions().last().unwrap().ncols(),
                        hierarchy.get_matrices().last().unwrap().nnz()
                    );
                    return true;
                }

                let p_transpose = new_partition.transpose();
                a_bar = &p_transpose * &(&a_bar * &new_partition);
                row_sums = &p_transpose * &row_sums;
                modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
            }
        }
    }
}

/// Takes a s.p.d matrix (mat), a vector that is near the nullspace of the matrix,
/// and a minimum coarsening factor for each level of the aggregation and provides a
/// hierarchy of partitions of the matrix.
pub fn modularity_matching(
    mat: Rc<CsrMatrix<f64>>,
    near_null: &'_ DVector<f64>,
    coarsening_factor: f64,
) -> Hierarchy {
    let (mut a_bar, mut row_sums, inverse_total) = build_weighted_matrix(&mat, &near_null);
    let mut modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
    let mut hierarchy = Hierarchy::new(mat);
    let mut partition_mat: Option<CsrMatrix<f64>> = None;
    let mut starting_vertex_count = modularity_mat.nrows() as f64;

    loop {
        let vertex_count = modularity_mat.nrows();

        match find_pairs(&modularity_mat, 1) {
            None => {
                assert_eq!(modularity_mat.nnz(), 0);
                /*
                if let Some(partition_mat) = partition_mat {
                    let cf = partition_mat.nrows() as f64 / partition_mat.ncols() as f64;
                    hierarchy.push(partition_mat, near_null);
                    trace!(
                        "added level because no pairs were found...! num vertices coarse: {} nnz: {} CF: {:.2}",
                        hierarchy.get_partitions().last().unwrap().ncols(),
                        hierarchy.get_matrices().last().unwrap().nnz(),
                        cf
                    );
                }
                */
                info!("Hierarchy constructed. Levels: {}", hierarchy.levels());
                for (i, mat) in hierarchy.get_matrices().iter().enumerate() {
                    info!(
                        "Level: {}\t Size: {}\t NNZ: {}",
                        i + 2,
                        mat.nrows(),
                        mat.nnz()
                    );
                }
                return hierarchy;
            }
            Some(pairs) => {
                let new_partition = build_partition_from_pairs(&pairs, vertex_count);
                let coarse_vertex_count = new_partition.ncols() as f64;

                if let Some(old_partition) = partition_mat {
                    partition_mat = Some(&old_partition * &new_partition);
                } else {
                    partition_mat = Some(new_partition.to_owned());
                }

                let p_transpose = new_partition.transpose();

                a_bar = &p_transpose * &(&a_bar * &new_partition);
                row_sums = spmm(&p_transpose, &row_sums);
                modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);

                if starting_vertex_count / coarse_vertex_count > coarsening_factor {
                    hierarchy.push(partition_mat.unwrap(), near_null);
                    trace!(
                        "added level! num vertices coarse: {} nnz: {}",
                        hierarchy.get_partitions().last().unwrap().ncols(),
                        hierarchy.get_matrices().last().unwrap().nnz()
                    );

                    partition_mat = None;
                    starting_vertex_count = coarse_vertex_count;
                }
            }
        }
    }
}

/// Takes a SPD matrix and a near null component and creates a weighted matrix with positive
/// row sums and a zero on the diagonal. This matrix can then be used for modularity based
/// matching to build the coarse systems. Returns a tuple with the weighted matrix, the row sums,
/// and the inverse of the total sum of the matrix.
fn build_weighted_matrix(
    mat: &CsrMatrix<f64>,
    near_null: &DVector<f64>,
) -> (CsrMatrix<f64>, DVector<f64>, f64) {
    let num_vertices = mat.nrows();
    let mut mat_bar = CooMatrix::new(num_vertices, num_vertices);
    for (i, j, val) in mat.triplet_iter().filter(|(i, j, _)| i != j) {
        let val_ij = val * (-near_null[i]) * near_null[j];
        mat_bar.push(i, j, val_ij);
    }
    let mat_bar = CsrMatrix::from(&mat_bar);
    let ones = DVector::from(vec![1.0; num_vertices]);

    let mut row_sums: DVector<f64> = &mat_bar * ones;

    //let mut counter = 0;
    //let mut total = 0.0;
    // TODO maybe just count how many rowsums are negative and by how much
    for sum in row_sums.iter_mut() {
        if *sum < 0.0 {
            //counter += 1;
            /*
            warn!(
                "sum was {sum} for row {i}. a_ii * w^2: {} w_i: {}. Setting to 0.0",
                mat.get_entry(i, i).unwrap().into_value() * near_null[i] * near_null[i],
                near_null[i]
            );
            */
            //total += *sum;
            *sum = 0.0
        }
    }

    /*
    if counter > 0 {
        warn!(
            "{} of {} rows had negative rowsums. Average negative: {:.1e}",
            counter,
            row_sums.nrows(),
            total / (counter as f64)
        );
    }
    */

    let total: f64 = row_sums.iter().sum();
    let inverse_total = 1.0 / total;

    (mat_bar, row_sums, inverse_total)
}

/// Constructs the modularity matrix for `mat` but only include the positive values.
/// The resulting matrix has a sparsity pattern that is the same or strictly more sparse
/// than the sparsity pattern of `mat`.
fn build_sparse_modularity_matrix(
    mat: &CsrMatrix<f64>,
    row_sums: &DVector<f64>,
    inverse_total: f64,
) -> CsrMatrix<f64> {
    let num_vertices = mat.nrows();
    let mut modularity_mat = CooMatrix::new(num_vertices, num_vertices);

    // NOTE: don't actually have to form this, can check if positive when looking
    // for pairs to merge which could save copying the entire matrix
    for (i, j, weight_ij) in mat.triplet_iter().filter(|(i, j, _)| i != j) {
        let modularity_ij = weight_ij - inverse_total * row_sums[i] * row_sums[j];
        if modularity_ij > 0.0 {
            modularity_mat.push(i, j, modularity_ij);
        }
    }

    CsrMatrix::from(&modularity_mat)
}

fn find_pairs(modularity_mat: &CsrMatrix<f64>, k_passes: usize) -> Option<Vec<(usize, usize)>> {
    let vertex_count = modularity_mat.nrows();
    let mut wants_to_merge: Vec<VecDeque<usize>> = modularity_mat
        .row_iter()
        .enumerate()
        .map(|(i, row)| {
            let mut possible_matches: Vec<(usize, f64)> = row
                .col_indices()
                .iter()
                .zip(row.values().iter())
                .filter(|(j, _)| i != **j)
                .map(|(j, weight)| (*j, *weight))
                .collect();
            // crashes on NaNs
            // NOTE: sorting here might be a bad idea... maybe get top (n?) instead?
            // probably best just to find largest every time and remove ones that are dead
            possible_matches.sort_by(|(_, w1), (_, w2)| w2.partial_cmp(w1).unwrap());
            possible_matches
                .into_iter()
                .map(|(j, _)| j)
                .collect::<VecDeque<usize>>()
        })
        .collect();

    if wants_to_merge.iter().all(|x| x.is_empty()) {
        return None;
    }

    let mut alive: Vec<bool> = wants_to_merge.iter().map(|x| !x.is_empty()).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(vertex_count / 2);

    for _ in 0..k_passes {
        for i in 0..vertex_count {
            if !alive[i] {
                continue;
            }
            if wants_to_merge[i].is_empty() {
                continue;
            }

            let j = loop {
                if let Some(j) = wants_to_merge[i].pop_front() {
                    if !alive[j] {
                        continue;
                    } else {
                        break Some(j);
                    }
                } else {
                    break None;
                }
            };

            if let Some(j) = j {
                if wants_to_merge[j].front() == Some(&i) {
                    pairs.push((i, j));
                    alive[j] = false;
                    alive[i] = false;
                }
            }
        }
    }

    if !pairs.is_empty() {
        Some(pairs)
    } else {
        None
    }
}

fn build_partition_from_pairs(pairs: &Vec<(usize, usize)>, vertex_count: usize) -> CsrMatrix<f64> {
    let mut not_merged_vertices: IndexSet<usize> = (0..vertex_count).collect();
    let pairs_count = pairs.len();
    let aggregate_count = vertex_count - pairs_count;
    //NOTE probably can go straight to csr now
    let mut partition_mat = CooMatrix::new(vertex_count, aggregate_count);

    for (aggregate, (i, j)) in pairs.into_iter().enumerate() {
        partition_mat.push(*i, aggregate, 1.0);
        partition_mat.push(*j, aggregate, 1.0);
        assert!(not_merged_vertices.remove(i));
        assert!(not_merged_vertices.remove(j));
    }

    for (aggregate, vertex) in not_merged_vertices.into_iter().enumerate() {
        partition_mat.push(vertex, aggregate + pairs_count, 1.0);
    }

    CsrMatrix::from(&partition_mat)
}

fn fix_bad_clusters(
    partition_mat: &mut CsrMatrix<f64>,
    near_null: &DVector<f64>,
    mat: &CsrMatrix<f64>,
) {
    // TODO we have all this data already so we shouldn't do it again....
    let (a_bar, row_sums, inverse_total) = build_weighted_matrix(mat, near_null);

    let abs = near_null.abs();
    let max = abs.max();
    // TODO study how this value changes performance of solver
    let threshold = 1e-15_f64;
    let almost_zero: IndexSet<_> = abs
        .iter()
        .enumerate()
        .filter(|(_, x)| threshold * max > **x)
        .map(|(i, _)| i)
        .collect();

    loop {
        let p_transpose = partition_mat.transpose();
        let mut new_a_bar = &p_transpose * &(&a_bar * &*partition_mat);
        let new_row_sums = &p_transpose * &row_sums;

        for (i, j, weight_ij) in new_a_bar.triplet_iter_mut() {
            *weight_ij -= inverse_total * new_row_sums[i] * new_row_sums[j];
        }

        let p_transpose = partition_mat.transpose();
        let clusters: Vec<IndexSet<_>> = p_transpose
            .row_iter()
            .map(|row| row.col_indices().iter().cloned().collect())
            .collect();
        let mut bad_clusters: IndexSet<usize> = clusters
            .iter()
            .enumerate()
            .filter(|(_, cluster)| cluster.is_subset(&almost_zero))
            .map(|(i, _)| i)
            .collect();

        if bad_clusters.is_empty() {
            return;
        }
        trace!("{} bad clusters remain", bad_clusters.len());

        let wants_to_merge: Vec<usize> = new_a_bar
            .row_iter()
            .enumerate()
            // TODO could filter here on only bad
            .map(|(i, row)| {
                row.col_indices()
                    .iter()
                    .zip(row.values().iter())
                    .filter(|(j, _)| i != **j)
                    .map(|(j, weight)| (*j, *weight))
                    .max_by(|(_, w1), (_, w2)| w2.partial_cmp(w1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let mut groups: Vec<IndexSet<usize>> = vec![];

        loop {
            if let Some(row) = bad_clusters.pop() {
                let wants_to_match = wants_to_merge[row];

                let mut found_group = false;
                for group in groups.iter_mut() {
                    if group.contains(&wants_to_match) {
                        group.insert(row);
                        found_group = true;
                        break;
                    }
                }
                if found_group == false {
                    groups.push(IndexSet::from([row, wants_to_match]));
                    bad_clusters.remove(&wants_to_match);
                }
            } else {
                break;
            }
        }
        let new_partition = build_partition_from_groups(&groups, partition_mat.ncols());

        *partition_mat = &*partition_mat * &new_partition;
    }
}

fn build_partition_from_groups(
    groups: &Vec<IndexSet<usize>>,
    vertex_count: usize,
) -> CsrMatrix<f64> {
    let mut not_merged_vertices: IndexSet<usize> = (0..vertex_count).collect();
    let groups_count = groups.len();
    let reduction: usize = groups.iter().map(|group| group.len() - 1).sum();
    let aggregate_count = vertex_count - reduction;
    //NOTE probably can go straight to csr now
    let mut partition_mat = CooMatrix::new(vertex_count, aggregate_count);

    for (aggregate, group) in groups.into_iter().enumerate() {
        for i in group {
            partition_mat.push(*i, aggregate, 1.0);
            assert!(not_merged_vertices.remove(i));
        }
    }

    for (aggregate, vertex) in not_merged_vertices.into_iter().enumerate() {
        partition_mat.push(vertex, aggregate + groups_count, 1.0);
    }

    CsrMatrix::from(&partition_mat)
}
