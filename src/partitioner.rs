use indexmap::IndexSet;
use ndarray::Array1;
use sprs::{CsMat, TriMat};
use std::collections::VecDeque;

//TODO test with P*1 = 1

/// Resulting object from running the modularity matching algorithm.
/// NOTE: Maybe don't store each matrix and just provide the P's.
#[derive(Clone)]
pub struct Hierarchy {
    partition_matrices: Vec<CsMat<f64>>,
    matrices: Vec<CsMat<f64>>,
}

impl Hierarchy {
    pub fn new(mat: CsMat<f64>) -> Self {
        Self {
            partition_matrices: vec![],
            matrices: vec![mat],
        }
    }

    /// Number of levels in the hierarchy. (Number of P matrices is one less)
    pub fn len(&self) -> usize {
        self.matrices.len()
    }

    /// Adds a level to the hierarchy.
    pub fn push(&mut self, partition_mat: CsMat<f64>) {
        let level = self.partition_matrices.len();
        let p_transpose = partition_mat.transpose_view().to_owned();
        let coarse_mat = &p_transpose * &(&self.matrices[level] * &partition_mat);
        self.matrices.push(coarse_mat);
        self.partition_matrices.push(partition_mat);
    }

    /// Get a single matrix from the hierarchy.
    pub fn get_matrix(&self, level: usize) -> &CsMat<f64> {
        &self.matrices[level]
    }

    /// Get a single P matrix from the hierarchy.
    pub fn get_partition(&self, level: usize) -> &CsMat<f64> {
        &self.partition_matrices[level]
    }

    /// Get a reference to the matrices Vec.
    pub fn get_matrices(&self) -> &Vec<CsMat<f64>> {
        &self.matrices
    }

    /// Get a reference to the P matrices Vec.
    pub fn get_partitions(&self) -> &Vec<CsMat<f64>> {
        &self.partition_matrices
    }
}

/// Takes a s.p.d matrix (mat), a vector that is near the nullspace of the matrix,
/// and a minimum coarsening factor for each level of the aggregation and provides a
/// hierarchy of partitions of the matrix.
pub fn modularity_matching(
    mat: CsMat<f64>,
    near_null: &Array1<f64>,
    coarsening_factor: f64,
) -> Hierarchy {
    let (mut a_bar, mut row_sums, inverse_total) = build_weighted_matrix(&mat, near_null);
    let mut modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
    let mut hierarchy = Hierarchy::new(mat);
    let mut partition_mat = None;
    let mut starting_vertex_count = modularity_mat.rows() as f64;

    loop {
        let vertex_count = modularity_mat.rows();

        match find_pairs(&modularity_mat, 10) {
            None => return hierarchy,
            Some(pairs) => {
                let pairs_count = pairs.len();
                trace!("num edges merged: {pairs_count}");
                let new_partition = build_partition_from_pairs(pairs, vertex_count);
                let coarse_vertex_count = new_partition.cols() as f64;

                if let Some(old_partition) = partition_mat {
                    partition_mat = Some(&old_partition * &new_partition);
                } else {
                    partition_mat = Some(new_partition.to_owned());
                }

                let p_transpose = new_partition.transpose_view().to_owned();

                a_bar = &p_transpose * &(&a_bar * &new_partition);
                row_sums = &p_transpose * &row_sums;
                modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);

                if starting_vertex_count / coarse_vertex_count > coarsening_factor {
                    trace!("added level! num vertices coarse: {}", new_partition.rows());
                    hierarchy.push(partition_mat.unwrap());
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
    mat: &CsMat<f64>,
    near_null: &Array1<f64>,
) -> (CsMat<f64>, Array1<f64>, f64) {
    let num_vertices = mat.rows();
    let mut mat_bar = TriMat::new((num_vertices, num_vertices));
    for (i, row_vec) in mat.outer_iterator().enumerate() {
        for (j, val) in row_vec.iter().filter(|(j, _)| *j != i) {
            let val_ij = val * -near_null[i] * near_null[j];
            mat_bar.add_triplet(i, j, val_ij);
        }
    }
    let mat_bar = mat_bar.to_csr::<usize>();

    let row_sums: Array1<f64> = mat_bar
        .outer_iterator()
        .map(|row_vec| row_vec.data().iter().sum())
        .collect();

    let total: f64 = row_sums.iter().sum();
    let inverse_total = 1.0 / total;

    (mat_bar, row_sums, inverse_total)
}

/// Constructs the modularity matrix for `mat` but only include the positive values.
/// The resulting matrix has a sparsity pattern that is the same or strictly more sparse
/// than the sparsity pattern of `mat`.
fn build_sparse_modularity_matrix(
    mat: &CsMat<f64>,
    row_sums: &Array1<f64>,
    inverse_total: f64,
) -> CsMat<f64> {
    let num_vertices = mat.rows();
    // NOTE: adding in row major order so maybe can use just CsMat,
    // but CsMat empty constructor doesn't take dimension so possibly bad?
    let mut modularity_mat = TriMat::new((num_vertices, num_vertices));

    for (i, row_vec) in mat.outer_iterator().enumerate() {
        for (j, weight_ij) in row_vec.iter() {
            let modularity_ij = weight_ij - inverse_total * row_sums[i] * row_sums[j];
            if modularity_ij > 0.0 {
                modularity_mat.add_triplet(i, j, modularity_ij);
            }
        }
    }

    modularity_mat.to_csr::<usize>()
}

fn find_pairs(modularity_mat: &CsMat<f64>, k_passes: usize) -> Option<Vec<(usize, usize)>> {
    let vertex_count = modularity_mat.rows();
    let mut wants_to_merge: Vec<VecDeque<usize>> = modularity_mat
        .outer_iterator()
        .enumerate()
        .map(|(i, row)| {
            let mut possible_matches: Vec<(usize, f64)> = row
                .iter()
                .filter(|(j, _)| i != *j)
                .map(|(j, weight)| (j, *weight))
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

    if wants_to_merge.iter().all(|x| x.len() == 0) {
        return None;
    }

    let mut alive: Vec<bool> = wants_to_merge.iter().map(|x| x.len() > 0).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(vertex_count / 2);

    for _ in 0..k_passes {
        for i in 0..vertex_count {
            if !alive[i] {
                continue;
            }
            if wants_to_merge[i].len() == 0 {
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

    if pairs.len() > 0 {
        Some(pairs)
    } else {
        None
    }
}

fn build_partition_from_pairs(pairs: Vec<(usize, usize)>, vertex_count: usize) -> CsMat<f64> {
    let mut not_merged_vertices: IndexSet<usize> = (0..vertex_count).collect();
    let pairs_count = pairs.len();
    let aggregate_count = vertex_count - pairs_count;
    let mut partition_mat = TriMat::new((vertex_count, aggregate_count));

    for (aggregate, (i, j)) in pairs.into_iter().enumerate() {
        partition_mat.add_triplet(i, aggregate, 1.0);
        partition_mat.add_triplet(j, aggregate, 1.0);
        assert!(not_merged_vertices.remove(&i));
        assert!(not_merged_vertices.remove(&j));
    }

    for (aggregate, vertex) in not_merged_vertices.into_iter().enumerate() {
        partition_mat.add_triplet(vertex, aggregate + pairs_count, 1.0);
    }

    partition_mat.to_csr::<usize>()
}

#[cfg(test)]
extern crate test_generator;

#[cfg(test)]
mod tests {
    use crate::{partitioner::modularity_matching, preconditioner::l1, solver::stationary};
    use ndarray::Array;
    use test_generator::test_resources;

    #[test_resources("test_matrices/*")]
    fn partition_times_ones_is_ones(mat_path: &str) {
        let mat = sprs::io::read_matrix_market::<f64, usize, _>(mat_path)
            .unwrap()
            .to_csr::<usize>();

        let ones = Array::from_vec(vec![1.0; mat.rows()]);
        let zeros = Array::from_vec(vec![0.0; mat.rows()]);

        let (near_null, _) = stationary(&mat, &zeros, &ones, 5, 10.0_f64.powi(-6), &l1(&mat));

        let hierarchy = modularity_matching(mat.clone(), &near_null, 2.0);

        for p in hierarchy.get_partitions().iter() {
            let ones = ndarray::Array::from_vec(vec![1.0; p.cols()]);
            let result = p * &ones;
            let inner_product = result.t().dot(&result);
            assert!((inner_product - result.len() as f64).abs() < 10e-6)
        }
    }
}
