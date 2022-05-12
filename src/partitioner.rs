use indexmap::IndexSet;
use nalgebra::base::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rand::prelude::*;
use rand::{distributions::Uniform, thread_rng};
use rayon::prelude::*;
use std::collections::VecDeque;

//TODO check pos rowsums in tests

/// Resulting object from running the modularity matching algorithm.
/// NOTE: Maybe don't store each matrix and just provide the P's.
#[derive(Clone)]
pub struct Hierarchy {
    partition_matrices: Vec<CsrMatrix<f64>>,
    matrices: Vec<CsrMatrix<f64>>,
}

impl Hierarchy {
    pub fn new(mat: CsrMatrix<f64>) -> Self {
        Self {
            partition_matrices: vec![],
            matrices: vec![mat],
        }
    }

    /// Number of levels in the hierarchy. (Number of P matrices is one less)
    pub fn len(&self) -> usize {
        self.matrices.len()
    }

    /// Check if the hierarchy has any levels
    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    /// Adds a level to the hierarchy.
    pub fn push(&mut self, partition_mat: CsrMatrix<f64>) {
        let level = self.partition_matrices.len();
        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(&self.matrices[level] * &partition_mat);
        self.matrices.push(coarse_mat);
        self.partition_matrices.push(partition_mat);
    }

    /// Get a single matrix from the hierarchy.
    pub fn get_matrix(&self, level: usize) -> &CsrMatrix<f64> {
        &self.matrices[level]
    }

    /// Get a single P matrix from the hierarchy.
    pub fn get_partition(&self, level: usize) -> &CsrMatrix<f64> {
        &self.partition_matrices[level]
    }

    /// Get a reference to the matrices Vec.
    pub fn get_matrices(&self) -> &Vec<CsrMatrix<f64>> {
        &self.matrices
    }

    /// Get a reference to the P matrices Vec.
    pub fn get_partitions(&self) -> &Vec<CsrMatrix<f64>> {
        &self.partition_matrices
    }
}

pub fn modularity_matching_no_copies(
    mat: CsrMatrix<f64>,
    mut near_null: DVector<f64>,
    mut row_sums: DVector<f64>,
    inverse_total: f64,
    coarsening_factor_per_level: f64,
    total_coarsening_factor: f64,
) -> Hierarchy {
    use std::sync::{Arc, Mutex};
    let k_passes = 50;
    let mut coarse_mat = mat.clone();
    let mut pairs = Vec::with_capacity(mat.nrows() / 3);
    let starting_vertex_count = mat.nrows() as f64;
    let mut level_starting_vertex_count = starting_vertex_count;
    let mut hierarchy = Hierarchy::new(mat);
    let mut partition_mat = None;

    loop {
        let vertex_count = coarse_mat.nrows();
        let mut wants_to_merge: Vec<Option<(usize, f64)>> = vec![None; vertex_count];
        let mut alive = vec![true; vertex_count];
        let alive_pairs = Arc::new(Mutex::new((&mut alive, &mut pairs)));

        for _ in 0..k_passes {
            {
                let (ref alive, _) = *alive_pairs.lock().unwrap();
                wants_to_merge
                    .par_iter_mut()
                    .enumerate()
                    .filter(|(i, _)| alive[*i])
                    .for_each(|(i, wants_to_merge)| {
                        let row = coarse_mat.row(i);
                        for (&j, val) in row
                            .col_indices()
                            .iter()
                            .zip(row.values())
                            .filter(|(&j, _)| i != j && alive[j])
                        {
                            let weight = val * -near_null[i] * near_null[j];
                            let modularity_ij = weight - inverse_total * row_sums[i] * row_sums[j];
                            if modularity_ij > 0.0 {
                                match wants_to_merge {
                                    None => *wants_to_merge = Some((j, modularity_ij)),
                                    Some((_, max)) => {
                                        if *max < modularity_ij {
                                            *wants_to_merge = Some((j, modularity_ij));
                                        }
                                    }
                                }
                            }
                        }
                    });
            }

            let start_pair_count = { alive_pairs.lock().unwrap().1.len() };
            wants_to_merge
                .par_iter()
                .enumerate()
                .filter(|(_, max)| max.is_some())
                .map(|(i, max)| (i, max.unwrap().0))
                .for_each(|(i, j)| {
                    if let Some((maybe_i, _)) = wants_to_merge[j] {
                        if maybe_i == i {
                            let (ref mut alive, ref mut pairs) = *alive_pairs.lock().unwrap();
                            if alive[i] && alive[j] {
                                pairs.push((i, j));
                                alive[i] = false;
                                alive[j] = false;
                            }
                        }
                    }
                });
            let end_pair_count = { alive_pairs.lock().unwrap().1.len() };
            if start_pair_count == end_pair_count {
                break;
            }
        }

        let (_, ref mut pairs) = *alive_pairs.lock().unwrap();
        // TODO also return somewhere if delta Q is too small per iteration
        if pairs.is_empty() {
            return hierarchy;
        } else {
            //trace!("In one iteration found {} pairs", pairs.len());
        }

        let new_partition = build_partition_from_pairs(&pairs, vertex_count);
        pairs.clear();
        let coarse_vertex_count = new_partition.ncols() as f64;

        if let Some(old_partition) = partition_mat {
            partition_mat = Some(old_partition * &new_partition);
        } else {
            partition_mat = Some(new_partition.to_owned());
        }

        let p_transpose = new_partition.transpose();

        coarse_mat = &p_transpose * (coarse_mat * &new_partition);
        row_sums = &p_transpose * row_sums;
        near_null = &p_transpose * near_null;

        let current_coarsening_factor = level_starting_vertex_count / coarse_vertex_count;
        if current_coarsening_factor > coarsening_factor_per_level {
            info!(
                "added level! num vertices coarse: {}",
                new_partition.ncols()
            );
            hierarchy.push(partition_mat.unwrap());
            partition_mat = None;
            level_starting_vertex_count = coarse_vertex_count;
            if starting_vertex_count / coarse_vertex_count > total_coarsening_factor {
                return hierarchy;
            }
        }
    }
}

fn try_row_sums(mat: &CsrMatrix<f64>, near_null: &mut DVector<f64>) -> Option<(DVector<f64>, f64)> {
    //no_zeroes(near_null);
    let num_vertices = mat.nrows();
    let mut row_sums: DVector<f64> = DVector::from(vec![0.0; num_vertices]);
    for (i, j, val) in mat.triplet_iter().filter(|(i, j, _)| i != j) {
        row_sums[i] += val * -near_null[i] * near_null[j];
    }

    let mut total = 0.0;
    for sum in row_sums.iter() {
        if *sum < 0.0 {
            return None;
        }
        total += sum;
    }

    let inverse_total = 1.0 / total;

    Some((row_sums, inverse_total))
}

/// Takes a s.p.d matrix (mat), a vector that is near the nullspace of the matrix,
/// and a minimum coarsening factor for each level of the aggregation and provides a
/// hierarchy of partitions of the matrix.
pub fn modularity_matching(
    mat: CsrMatrix<f64>,
    near_null: &DVector<f64>,
    coarsening_factor: f64,
) -> Hierarchy {
    let (mut a_bar, mut row_sums, inverse_total) = build_weighted_matrix(&mat, &near_null);
    let mut modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
    let mut hierarchy = Hierarchy::new(mat);
    let mut partition_mat: Option<CsrMatrix<f64>> = None;
    let mut starting_vertex_count = modularity_mat.nrows() as f64;

    loop {
        let vertex_count = modularity_mat.nrows();

        match find_pairs(&modularity_mat, 30) {
            None => {
                if let Some(p) = partition_mat {
                    trace!("added level! num vertices coarse: {}", p.ncols());
                    hierarchy.push(p);
                }

                info!("Levels: {}", hierarchy.len());
                return hierarchy;
            }
            Some(pairs) => {
                /*
                let pairs_count = pairs.len();
                if pairs_count < 3 {
                    info!("Levels: {}", hierarchy.len());
                    return hierarchy;
                }
                */
                //trace!("num edges merged: {pairs_count}");
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
                    trace!(
                        "added level! num vertices coarse: {}",
                        new_partition.ncols()
                    );
                    hierarchy.push(partition_mat.unwrap());
                    if coarse_vertex_count < 16.0 {
                        info!("Levels: {}", hierarchy.len());
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

    for (i, sum) in row_sums.iter_mut().enumerate() {
        if *sum < 0.0 {
            /*
            warn!(
                "sum was {sum} for row {i}. a_ii * w^2: {} w_i: {}. Setting to 0.0",
                mat.get_entry(i, i).unwrap().into_value() * near_null[i] * near_null[i],
                near_null[i]
            );
            */
            *sum = 0.0
        }
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
    // NOTE: adding in row major order so maybe can use just CsrMatrix,
    // but CsrMatrix empty constructor doesn't take dimension so possibly bad?
    let mut modularity_mat = CooMatrix::new(num_vertices, num_vertices);

    // NOTE: don't actually have to form this, can check if positive when looking
    // for pairs to merge which could save copying the entire matrix
    for (i, j, weight_ij) in mat.triplet_iter() {
        let modularity_ij = weight_ij - inverse_total * row_sums[i] * row_sums[j];
        //if modularity_ij > 0.0 {
        modularity_mat.push(i, j, modularity_ij);
        //}
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
                alive[i] = false;
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
