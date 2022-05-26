use indexmap::IndexSet;
use nalgebra::base::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rayon::prelude::*;
use sprs::CsMat;
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
        } else {
            fine_mat = self.matrices.last().unwrap();
        }

        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &partition_mat);
        self.matrices.push(coarse_mat);
        self.partition_matrices.push(partition_mat);
        self.interpolation_matrices.push(p_transpose);
    }

    pub fn push_sprs(&mut self, partition_mat: CsMat<f64>, near_null: &DVector<f64>) {
        let (n_rows, n_cols) = partition_mat.shape();
        let (indptr, indices, data) = partition_mat.into_raw_storage();
        let p = CsrMatrix::try_from_csr_data(n_rows, n_cols, indptr, indices, data).unwrap();
        self.push(p, near_null);
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

pub fn parallel_modularity_matching<'a>(
    mat: &'a CsrMatrix<f64>,
    near_null: &'_ DVector<f64>,
    coarsening_factor: f64,
) -> Hierarchy<'a> {
    let dim = mat.nrows();
    let mut hierarchy = Hierarchy::new(mat);
    let triplets: Vec<(usize, usize, f64)> = mat
        .triplet_iter()
        .filter(|(i, j, _)| i != j)
        .map(|(i, j, v)| (i, j, *v))
        .collect();

    let (mut mat_bar, mut row_sums, inverse_total) =
        parallel_build_weighted_matrix(dim, triplets, &near_null);

    let mut starting_vertex_count = dim as f64;
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(dim / 2);
    let mut partition_mat: CsMat<f64> = CsMat::eye(dim);

    loop {
        parallel_find_pairs(&mut pairs, &mat_bar, &row_sums, inverse_total);
        if pairs.is_empty() {
            info!("Levels: {}", hierarchy.levels());
            return hierarchy;
        }

        let new_partition = parallel_build_partition_from_pairs(&pairs, mat_bar.rows());
        partition_mat = sprs::smmp::mul_csr_csr(partition_mat.view(), new_partition.view());

        let a_p: CsMat<f64> = sprs::smmp::mul_csr_csr(mat_bar.view(), new_partition.view());
        let p_t = new_partition.transpose_view().to_owned().into_csr();
        mat_bar = sprs::smmp::mul_csr_csr(p_t.view(), a_p.view());
        row_sums = par_row_sums(&mat_bar);

        let coarse_vertex_count = partition_mat.cols() as f64;
        if starting_vertex_count / coarse_vertex_count > coarsening_factor {
            hierarchy.push_sprs(partition_mat.clone(), &near_null);
            trace!(
                "added level! num vertices coarse: {} nnz: {}",
                hierarchy.get_partitions().last().unwrap().ncols(),
                hierarchy.get_matrices().last().unwrap().nnz()
            );
            if coarse_vertex_count < 1000.0 {
                info!("Levels: {}", hierarchy.levels());
                return hierarchy;
            }
            partition_mat = CsMat::eye(partition_mat.cols());
            starting_vertex_count = coarse_vertex_count;
        }
    }
}

fn par_row_sums(mat: &CsMat<f64>) -> DVector<f64> {
    let dim = mat.rows();
    let mut row_sums: DVector<f64> = DVector::from(vec![0.0; dim]);
    row_sums
        .as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| {
            *val = mat.outer_view(i).unwrap().iter().map(|(_, v)| *v).sum();
            if *val < 0.0 {
                *val = 0.0
            }
        });

    row_sums
}

fn parallel_build_partition_from_pairs(
    pairs: &Vec<(usize, usize)>,
    vertex_count: usize,
) -> CsMat<f64> {
    let mut not_merged_vertices: IndexSet<usize> = (0..vertex_count).collect();
    let pairs_count = pairs.len();
    let aggregate_count = vertex_count - pairs_count;
    //NOTE probably can go straight to csr now
    let mut partition_mat = sprs::TriMat::new((vertex_count, aggregate_count));

    for (aggregate, (i, j)) in pairs.into_iter().enumerate() {
        partition_mat.add_triplet(*i, aggregate, 1.0);
        partition_mat.add_triplet(*j, aggregate, 1.0);
        assert!(not_merged_vertices.remove(i));
        assert!(not_merged_vertices.remove(j));
    }

    for (aggregate, vertex) in not_merged_vertices.into_iter().enumerate() {
        partition_mat.add_triplet(vertex, aggregate + pairs_count, 1.0);
    }

    partition_mat.to_csr()
}

fn parallel_find_pairs(
    pairs: &mut Vec<(usize, usize)>,
    mat_bar: &CsMat<f64>,
    row_sums: &DVector<f64>,
    inverse_total: f64,
) {
    pairs.clear();
    let dim = mat_bar.rows();

    let wants_to_merge: Vec<Option<usize>> = (0..dim)
        //.into_par_iter()
        .map(|i| {
            mat_bar
                .outer_view(i)
                .unwrap()
                .iter()
                .filter(|(j, _)| i != *j)
                .fold(None, |acc, (j, weight_ij)| {
                    let modularity_ij = weight_ij - inverse_total * row_sums[i] * row_sums[j];
                    if modularity_ij > 0.0 {
                        match acc {
                            None => Some((j, modularity_ij)),
                            Some((old_j, old_modularity_ij)) => {
                                if modularity_ij > old_modularity_ij {
                                    Some((j, modularity_ij))
                                } else {
                                    Some((old_j, old_modularity_ij))
                                }
                            }
                        }
                    } else {
                        acc
                    }
                })
                .map(|x| x.0)
        })
        .collect();

    // TODO maybe make par also
    for (i, maybe_j) in wants_to_merge.iter().enumerate() {
        if let Some(j) = *maybe_j {
            if Some(i) == wants_to_merge[j] && i < j {
                pairs.push((i, j));
            }
        }
    }
}

fn parallel_build_weighted_matrix(
    dim: usize,
    mut triplets: Vec<(usize, usize, f64)>,
    near_null: &DVector<f64>,
) -> (CsMat<f64>, DVector<f64>, f64) {
    triplets.par_iter_mut().for_each(|(i, j, val)| {
        *val *= -near_null[*i] * near_null[*j];
    });

    //TODO this breaks because some rows are empty in FEM matrices
    let mut mat_bar = sprs::TriMat::new((dim, dim));
    for (i, j, val) in triplets.into_iter() {
        mat_bar.add_triplet(i, j, val);
    }

    let mat_bar: CsMat<f64> = mat_bar.to_csr();

    let mut row_sums: DVector<f64> = DVector::from(vec![0.0; dim]);
    let counter = std::sync::atomic::AtomicUsize::new(0);
    use std::sync::atomic::Ordering::Relaxed;
    row_sums
        .as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| {
            *val = mat_bar.outer_view(i).unwrap().iter().map(|(_, v)| *v).sum();
            if *val < 0.0 {
                counter.fetch_add(1, Relaxed);
                *val = 0.0;
            }
        });

    let counter = counter.into_inner();
    if counter > 0 {
        warn!("{} rows had negative rowsums", counter,);
    }

    let total: f64 = row_sums.as_slice().par_iter().sum();
    let inverse_total = 1.0 / total;

    (mat_bar, row_sums, inverse_total)
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
                /*
                if let Some(p) = partition_mat {
                    hierarchy.push(p);
                    trace!(
                        "added level! num vertices coarse: {} nnz: {}",
                        hierarchy.get_partitions().last().unwrap().ncols(),
                        hierarchy.get_matrices().last().unwrap().nnz()
                    );
                }
                */

                info!("Levels: {}", hierarchy.levels());
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
            "{} rows had negative rowsums. Average negative: {}",
            counter,
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
