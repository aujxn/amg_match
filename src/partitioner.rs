//! This module contains methods that partitions the matrix into hierarchies.
//! This could potentially be moved to a seperate crate, since I have copied
//! this code into other projects as well.

use metis::Graph;
use nalgebra::base::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rayon::slice::ParallelSliceMut;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

/// A non-overlapping partition of square matrix (or 'graph').
pub struct Partition {
    pub mat: Arc<CsrMatrix<f64>>,
    pub node_to_agg: Vec<usize>,
    pub agg_to_node: Vec<BTreeSet<usize>>,
}

/// Use Metis C API bindings to partition a symmetric matrix (undirected graph). Here we min-max map the modularity
/// weights into quantized integer values as required by Metis. Do not try to partition large
/// graphs into many (thousdands) of parts or Metis will crash. Either apply this recursively with
/// some approach or use the greedy matching algorithm. Also, if your matrix has more than
/// `i32::MAX` non-zeros (graph edges) Metis will crash.
pub fn metis_n(near_null: &'_ DVector<f64>, mat: Arc<CsrMatrix<f64>>, n_parts: usize) -> Partition {
    let (a_bar, row_sums, inverse_total) = build_weighted_matrix(&mat, &near_null);
    let mut modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
    modularity_mat = modularity_mat.filter(|_, _, val| *val > 0.0);

    let max = modularity_mat
        .values()
        .iter()
        .fold(0.0, |acc, x| if acc > *x { acc } else { *x });
    let min = modularity_mat
        .values()
        .iter()
        .fold(9999999.0, |acc, x| if acc < *x { acc } else { *x });
    let dif = max - min;

    let xadj: Vec<i32> = modularity_mat
        .row_offsets()
        .iter()
        .map(|i| *i as i32)
        .collect();
    let adjncy: Vec<i32> = modularity_mat
        .col_indices()
        .iter()
        .map(|j| *j as i32)
        .collect();

    let weigts: Vec<i32> = modularity_mat
        .values()
        .iter()
        .map(|val| (1e6 * (val - min) / dif).ceil() as i32 + 1)
        .collect();

    let graph = Graph::new(1_i32, n_parts as i32, &xadj, &adjncy).unwrap();
    let graph = graph.set_adjwgt(&weigts);
    let mut partition = vec![0_i32; modularity_mat.nrows()];
    graph.part_kway(&mut partition).unwrap();

    let mut node_to_agg = Vec::with_capacity(mat.nrows());
    let mut agg_to_node = vec![BTreeSet::new(); n_parts];

    for (i, agg) in partition.into_iter().map(|agg| agg as usize).enumerate() {
        node_to_agg.push(agg);
        agg_to_node[agg].insert(i);
    }

    Partition {
        mat,
        node_to_agg,
        agg_to_node,
    }
}

/// Partition a symmetric matrix (undirected graph) with greedy matching of modularity weights.
/// Performs pairwise matching until the desired coarsening factor is reached:
///
/// `fine_size / coarse_size > coarsening_factor`
///
/// To achieve balance aggregates, choose a `max_agg_size = ceil(coarsening_factor)` or `ceil(coarsening_factor) + 1`.
pub fn modularity_matching_partition(
    mat: Arc<CsrMatrix<f64>>,
    near_null: &DVector<f64>,
    coarsening_factor: f64,
    max_agg_size: Option<usize>,
) -> Partition {
    let ndofs = mat.nrows();
    let fine_ndofs = ndofs as f64;
    let mut aggs = (0..ndofs)
        .map(|i| {
            let mut agg = BTreeSet::new();
            agg.insert(i);
            agg
        })
        .collect();

    let (mut a_bar, mut row_sums, inverse_total) = build_weighted_matrix_coosym(&mat, &near_null);

    loop {
        let matches = greedy_matching(
            &mut a_bar,
            &mut row_sums,
            &mut aggs,
            inverse_total,
            max_agg_size,
        );
        let coarse_ndofs = aggs.len() as f64;
        let current_cf = fine_ndofs / coarse_ndofs;

        if matches == 0 {
            trace!("Greedy partitioner terminated because no more matches are possible. target cf: {:.2} achieved: {:.2}", coarsening_factor, current_cf);
            break;
        }
        if current_cf > coarsening_factor {
            trace!(
                "Greedy partitioner achieved target cf of {:.2} with actual cf {:.2}",
                coarsening_factor,
                current_cf
            );
            break;
        }
    }

    let mut node_to_agg = vec![0; ndofs];
    for (agg_id, agg) in aggs.iter().enumerate() {
        for idx in agg.iter().cloned() {
            node_to_agg[idx] = agg_id;
        }
    }

    Partition {
        mat,
        node_to_agg,
        agg_to_node: aggs,
    }
}

fn build_weighted_matrix_coosym(
    mat: &CsrMatrix<f64>,
    near_null: &DVector<f64>,
) -> (Vec<(usize, usize, f64)>, DVector<f64>, f64) {
    let mut row_sums: DVector<f64> = DVector::from_element(mat.nrows(), 0.0);
    let mut total: f64 = 0.0;
    let mut mat_bar = Vec::new();

    for (i, j, val) in mat.triplet_iter().filter(|(i, j, _)| i < j) {
        let strength_ij = -val * near_null[i] * near_null[j];
        mat_bar.push((i, j, strength_ij));
        row_sums[i] += strength_ij;
        row_sums[j] += strength_ij;
        total += 2.0 * strength_ij;
    }

    // Some sanity checks since everthing here on is based on the assumption that
    // $Aw \approx 0$ giving that the row-sums of $\bar{A}$ are positive. Things close to 0 and
    // negative are fine, just set them to 0, but output a warning with how close.
    let mut counter = 0;
    let mut total_bad = 0.0;
    let mut min = 0.0;

    for sum in row_sums.iter_mut() {
        if *sum < 0.0 {
            counter += 1;
            if *sum < min {
                min = *sum;
            }
            total_bad += *sum;
            *sum = 0.0;
        }
    }

    if counter > 0 {
        warn!(
            "{} of {} rows had negative rowsums. Average negative: {:.1e}, worst negative: {:.1e}",
            counter,
            row_sums.nrows(),
            total_bad / (counter as f64),
            min
        );
    }

    (mat_bar, row_sums, 1.0 / (total - total_bad))
}

fn greedy_matching(
    a_bar: &mut Vec<(usize, usize, f64)>,
    row_sums: &mut DVector<f64>,
    aggs: &mut Vec<BTreeSet<usize>>,
    inverse_total: f64,
    max_agg_size: Option<usize>,
) -> usize {
    let vertex_count = row_sums.len();

    //trace!("Starting matching with {} ndofs", vertex_count);
    //trace!("Starting matching with {} aggs", aggs.len());
    let mut wants_to_merge: Vec<(usize, usize, f64)> = a_bar
        .iter()
        .copied()
        .filter(|(i, j, w)| {
            if let Some(max_size) = max_agg_size {
                let agg_i_size = aggs[*i].len();
                let agg_j_size = aggs[*j].len();
                *w > 0.0 && agg_j_size + agg_i_size <= max_size
            } else {
                *w > 0.0
            }
        })
        .map(|(i, j, w)| {
            let modularity_ij = w - inverse_total * row_sums[i] * row_sums[j];
            (i, j, modularity_ij)
        })
        .collect();

    wants_to_merge.par_sort_by(|(_, _, w1), (_, _, w2)| w1.partial_cmp(w2).unwrap());

    if wants_to_merge.is_empty() {
        return 0;
    }

    let mut alive: HashSet<usize> = (0..vertex_count).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(vertex_count / 2);

    loop {
        match wants_to_merge.pop() {
            None => break,
            Some((i, j, _w)) => {
                if alive.contains(&i) && alive.contains(&j) {
                    alive.remove(&i);
                    alive.remove(&j);
                    pairs.push((i, j));
                }
            }
        }
    }

    assert!(!pairs.is_empty());

    let matched = pairs.len();
    //trace!("matched {} pairs", matched);
    let new_agg_count = aggs.len() - matched;
    //trace!("new aggs count {}", new_agg_count);
    let mut new_aggs = Vec::with_capacity(new_agg_count);
    let mut mapping = vec![0_usize; aggs.len()];
    let mut new_row_sums = DVector::from_element(new_agg_count, 0.0);

    let mut agg_id = 0;
    for (i, j) in pairs.into_iter() {
        mapping[i] = agg_id;
        mapping[j] = agg_id;

        aggs.push(BTreeSet::new());
        let mut a = aggs.swap_remove(i);
        aggs.push(BTreeSet::new());
        let b = aggs.swap_remove(j);

        a.extend(&b);
        new_aggs.push(a);
        agg_id += 1;
    }
    for i in alive {
        mapping[i] = agg_id;
        agg_id += 1;
        aggs.push(BTreeSet::new());
        new_aggs.push(aggs.swap_remove(i));
    }
    assert_eq!(agg_id, new_agg_count);
    for agg in aggs.into_iter() {
        assert!(agg.is_empty());
    }
    *aggs = new_aggs;

    *a_bar = a_bar
        .iter()
        .cloned()
        .map(|(i, j, v)| (mapping[i], mapping[j], v))
        .filter(|(i, j, _)| *i != *j)
        .map(|(i, j, v)| if i > j { (j, i, v) } else { (i, j, v) })
        .collect();

    a_bar.sort_by(|a, b| {
        if a.0 != b.0 {
            a.0.cmp(&b.0)
        } else {
            a.1.cmp(&b.1)
        }
    });
    /*
    for (i, j, v) in a_bar.iter().take(50) {
        println!("{}, {}, {:.2}", i, j, v);
    }
    */
    let mut write_idx = 0;
    let mut current_i = a_bar[0].0;
    let mut current_j = a_bar[0].1;
    for read_idx in 1..a_bar.len() {
        let next_i = a_bar[read_idx].0;
        let next_j = a_bar[read_idx].1;
        let next_val = a_bar[read_idx].2;
        if current_i == next_i && current_j == next_j {
            a_bar[write_idx].2 += next_val;
        } else {
            write_idx += 1;
            current_i = next_i;
            current_j = next_j;
            a_bar[write_idx] = (next_i, next_j, next_val);
        }
    }
    /*
    for (i, j, v) in a_bar.iter().take(50) {
        println!("{}, {}, {:.2}", i, j, v);
    }
    let max_idx = mapping.iter().max().unwrap();
    trace!("Max coarse idx in mapping: {}", max_idx);
    */

    for (fine_i, coarse_i) in mapping.iter().copied().enumerate() {
        new_row_sums[coarse_i] += row_sums[fine_i];
    }
    *row_sums = new_row_sums;

    matched
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
    let total: f64 = row_sums.iter().sum();
    let inverse_total = 1.0 / total;

    // Some sanity checks since everthing here on is based on the assumption that
    // $Aw \approx 0$ giving that the row-sums of $\bar{A}$ are positive. Things close to 0 and
    // negative are fine, just set them to 0, but output a warning with how close.
    let mut counter = 0;
    let mut total = 0.0;
    let mut min = 0.0;

    for sum in row_sums.iter_mut() {
        if *sum < 0.0 {
            counter += 1;
            if *sum < min {
                min = *sum;
            }
            total += *sum;
            *sum = 0.0;
        }
    }

    if counter > 0 {
        warn!(
            "{} of {} rows had negative rowsums. Average negative: {:.1e}, worst negative: {:.1e}",
            counter,
            row_sums.nrows(),
            total / (counter as f64),
            min
        );
    }

    (mat_bar, row_sums, inverse_total)
}

/// Constructs the modularity matrix for `mat` but restricted to the sparsity pattern of `mat`.
fn build_sparse_modularity_matrix(
    mat: &CsrMatrix<f64>,
    row_sums: &DVector<f64>,
    inverse_total: f64,
) -> CsrMatrix<f64> {
    let num_vertices = mat.nrows();
    let mut modularity_mat = CooMatrix::new(num_vertices, num_vertices);

    // NOTE: don't actually have to form this, can check if positive when looking
    // for pairs to merge which could save copying the entire matrix. For now this
    // is more simple and premature optimization is the root of all evil.
    for (i, j, weight_ij) in mat.triplet_iter().filter(|(i, j, w)| i != j && **w > 0.0) {
        let modularity_ij = weight_ij - inverse_total * row_sums[i] * row_sums[j];
        /*
        if modularity_ij > 0.0 {
            modularity_mat.push(i, j, modularity_ij);
        }
        */
        modularity_mat.push(i, j, modularity_ij);
    }

    CsrMatrix::from(&modularity_mat)
}

/*
pub fn modularity_matching_add_level(
    near_null: &'_ DVector<f64>,
    coarsening_factor: f64,
    hierarchy: &mut Hierarchy,
    interpolation_type: InterpolationType,
    //save: bool,
) -> DVector<f64> {
    hierarchy.near_nulls.push(near_null.clone());
    let mat = hierarchy
        .get_matrices()
        .last()
        .map_or(hierarchy.get_mat(0), |m| m.clone());
    let (mut a_bar, mut row_sums, inverse_total) = build_weighted_matrix(&mat, &near_null);
    let mut modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);

    /*
    if save {
        //let path = format!("strength_lvl{}.mtx", hierarchy.levels());
        //let path_pos = format!("strength_lvl{}_pos.mtx", hierarchy.levels());
        let mut only_pos = CooMatrix::new(a_bar.nrows(), a_bar.ncols());
        for (i, j, v) in a_bar.triplet_iter() {
            if i != j && *v > 0.0 {
                only_pos.push(i, j, *v)
            }
        }
        a_bar = CsrMatrix::from(&only_pos);
        //let _ = nalgebra_sparse::io::save_to_matrix_market_file(&a_bar, path);
        //let _ = nalgebra_sparse::io::save_to_matrix_market_file(&only_pos, path_pos);
    }
    */

    let dim = mat.nrows();
    let mut partition_mat = CsrMatrix::identity(dim);
    let starting_vertex_count = dim as f64;
    let mut agg_sizes = vec![1; dim];
    let max_agg_size = coarsening_factor.ceil() as usize + 1;

    loop {
        let vertex_count = modularity_mat.nrows();

        match find_pairs_force_all(&modularity_mat, &agg_sizes, max_agg_size) {
            None => {
                return hierarchy.add_level(partition_mat, near_null, interpolation_type);
            }
            Some(pairs) => {
                let new_partition = build_partition_from_pairs(&pairs, vertex_count);
                let coarse_vertex_count = new_partition.ncols() as f64;
                partition_mat = &partition_mat * &new_partition;

                let p_transpose = partition_mat.transpose();
                agg_sizes = p_transpose
                    .row_iter()
                    .map(|row| row.values().len())
                    .collect();

                if starting_vertex_count / coarse_vertex_count > coarsening_factor {
                    return hierarchy.add_level(partition_mat, near_null, interpolation_type);
                }

                let p_transpose = new_partition.transpose();
                a_bar = &p_transpose * &(&a_bar * &new_partition);
                row_sums = &p_transpose * &row_sums;
                modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);
            }
        }
    }
}

fn find_pairs_force_all(
    modularity_mat: &CsrMatrix<f64>,
    agg_sizes: &Vec<usize>,
    max_agg_size: usize,
) -> Option<Vec<(usize, usize)>> {
    let vertex_count = modularity_mat.nrows();
    let mut wants_to_merge: Vec<(usize, usize, f64)> = modularity_mat
        .triplet_iter()
        .filter(|triplet| triplet.0 > triplet.1 && *triplet.2 > 0.0)
        .map(|(i, j, w)| (i, j, *w))
        .collect();

    // crashes on NaNs...
    wants_to_merge.par_sort_by(|(_, _, w1), (_, _, w2)| w2.partial_cmp(w1).unwrap());
    let mut wants_to_merge: VecDeque<(usize, usize, f64)> = VecDeque::from(wants_to_merge);

    if wants_to_merge.is_empty() {
        return None;
    }

    let mut alive: Vec<bool> = vec![true; vertex_count];
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(vertex_count / 2);

    loop {
        match wants_to_merge.pop_front() {
            None => break,
            Some((i, j, _w)) => {
                if alive[i] && alive[j] && agg_sizes[i] + agg_sizes[j] <= max_agg_size {
                    alive[i] = false;
                    alive[j] = false;
                    pairs.push((i, j));
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

// TODO needs some work to be efficient and create numerically stable hierarchies (max agg size,
// more greed tuning)
/*
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
                    }
                    break Some(j);
                }
                break None;
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
*/

fn build_partition_from_pairs(pairs: &Vec<(usize, usize)>, vertex_count: usize) -> CsrMatrix<f64> {
    let mut not_merged_vertices: BTreeSet<usize> = (0..vertex_count).collect();
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

/*
fn build_partition_from_groups(
    groups: &Vec<BTreeSet<usize>>,
    vertex_count: usize,
) -> CsrMatrix<f64> {
    let mut not_merged_vertices: BTreeSet<usize> = (0..vertex_count).collect();
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
*/
*/
