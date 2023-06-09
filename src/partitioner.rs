//! This module contains methods that partitions the matrix into hierarchies.
//! This could potentially be moved to a seperate crate, since I have copied
//! this code into other projects as well.

use indexmap::IndexSet;
use nalgebra::base::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rayon::prelude::*;
use std::collections::VecDeque;

//TODO bring back tests you deleted when preconditioner refactor happened
//     also, stop copying the fine mat into the hierarchy

/// Resulting object from running the modularity matching algorithm.
/// NOTE: Maybe don't store each matrix and just provide the P's.
#[derive(Clone)]
pub struct Hierarchy<'a> {
    mat: &'a CsrMatrix<f64>,
    partition_matrices: Vec<CsrMatrix<f64>>,
    interpolation_matrices: Vec<CsrMatrix<f64>>,
    matrices: Vec<CsrMatrix<f64>>,
}

impl<'a> Hierarchy<'a> {
    pub fn new(mat: &'a CsrMatrix<f64>) -> Self {
        Self {
            mat,
            partition_matrices: Vec::new(),
            interpolation_matrices: Vec::new(),
            matrices: Vec::new(),
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
        let fine_mat;
        let mut partition_mat = partition_mat.clone();
        if self.matrices.is_empty() {
            fine_mat = self.mat;
            partition_mat
                .triplet_iter_mut()
                .for_each(|(i, _, w)| *w *= near_null[i]);

            // TODO test product of all Ps with coarse ones to get w
            let ones = DVector::from(vec![1.0; partition_mat.ncols()]);
            let maybe_w = &partition_mat * &ones;
            for (w_i, maybe_w_i) in near_null.iter().zip(maybe_w.iter()) {
                assert!((maybe_w_i - w_i).abs() < 1e-10);
            }
        } else {
            fine_mat = self.matrices.last().unwrap();
        }

        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &partition_mat);
        self.matrices.push(coarse_mat);
        self.partition_matrices.push(partition_mat);
        self.interpolation_matrices.push(p_transpose);
    }

    pub fn push_p_scaled_by_w(
        &mut self,
        mut partition_mat: CsrMatrix<f64>,
        near_null: &DVector<f64>,
    ) {
        let fine_mat;
        if self.matrices.is_empty() {
            fine_mat = self.mat;
        } else {
            fine_mat = self.matrices.last().unwrap();
        }
        partition_mat
            .triplet_iter_mut()
            .for_each(|(i, _, w)| *w *= near_null[i]);

        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &partition_mat);
        self.matrices.push(coarse_mat);
        self.partition_matrices.push(partition_mat);
        self.interpolation_matrices.push(p_transpose);
    }

    /// Get a single P matrix from the hierarchy.
    pub fn get_partition(&self, level: usize) -> &CsrMatrix<f64> {
        &self.partition_matrices[level]
    }

    /// Get a reference to the matrices Vec.
    pub fn get_matrices(&self) -> &[CsrMatrix<f64>] {
        &self.matrices
    }

    /// Get a reference to the P matrices Vec.
    pub fn get_partitions(&self) -> &Vec<CsrMatrix<f64>> {
        &self.partition_matrices
    }

    /// Get a reference to the P matrices Vec.
    pub fn get_interpolations(&self) -> &Vec<CsrMatrix<f64>> {
        &self.interpolation_matrices
    }
}

impl<'a> std::ops::Index<usize> for Hierarchy<'a> {
    type Output = CsrMatrix<f64>;

    fn index(&self, level: usize) -> &Self::Output {
        if level == 0 {
            self.mat
        } else {
            &self.matrices[level - 1]
        }
    }
}

pub fn modularity_matching_add_level<'a>(
    near_null: &'_ DVector<f64>,
    coarsening_factor: f64,
    hierarchy: &mut Hierarchy,
) -> bool {
    let mat = hierarchy.get_matrices().last().unwrap_or(&hierarchy[0]);
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
                    hierarchy.push_p_scaled_by_w(partition_mat, near_null);
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
pub fn modularity_matching<'a>(
    mat: &'a CsrMatrix<f64>,
    near_null: &'_ DVector<f64>,
    coarsening_factor: f64,
) -> Hierarchy<'a> {
    let (mut a_bar, mut row_sums, inverse_total) = build_weighted_matrix(&mat, &near_null);
    let mut modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
    let mut hierarchy = Hierarchy::new(mat);
    let mut partition_mat: Option<CsrMatrix<f64>> = None;
    let mut starting_vertex_count = modularity_mat.nrows() as f64;

    loop {
        let vertex_count = modularity_mat.nrows();

        match find_pairs(&modularity_mat, 1) {
            None => {
                info!("Levels: {}", hierarchy.levels());
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
                row_sums = &p_transpose * &row_sums;
                modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);

                if starting_vertex_count / coarse_vertex_count > coarsening_factor {
                    hierarchy.push(partition_mat.unwrap(), near_null);
                    trace!(
                        "added level! num vertices coarse: {} nnz: {}",
                        hierarchy.get_partitions().last().unwrap().ncols(),
                        hierarchy.get_matrices().last().unwrap().nnz()
                    );
                    if coarse_vertex_count < 1000.0 {
                        info!("Levels: {}", hierarchy.levels());
                        return hierarchy;
                    }
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
        let val_ij = val * -near_null[i] * near_null[j];
        mat_bar.push(i, j, val_ij);
    }
    let mat_bar = CsrMatrix::from(&mat_bar);
    let ones = DVector::from(vec![1.0; num_vertices]);

    let mut row_sums: DVector<f64> = &mat_bar * ones;

    let mut counter = 0;
    let mut total = 0.0;
    // TODO maybe just count how many rowsums are negative and by how much
    for sum in row_sums.iter_mut() {
        if *sum < 0.0 {
            counter += 1;
            /*
            warn!(
                "sum was {sum} for row {i}. a_ii * w^2: {} w_i: {}. Setting to 0.0",
                mat.get_entry(i, i).unwrap().into_value() * near_null[i] * near_null[i],
                near_null[i]
            );
            */
            total += *sum;
            *sum = 0.0
        }
    }

    if counter > 0 {
        warn!(
            "{} of {} rows had negative rowsums. Average negative: {:.3e}",
            counter,
            row_sums.nrows(),
            total / (counter as f64)
        );
    }

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
    for (i, j, weight_ij) in mat.triplet_iter() {
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
