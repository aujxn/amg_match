//! This module contains methods that partitions the matrix into hierarchies.
//! This could potentially be moved to a seperate crate, since I have copied
//! this code into other projects as well.

use metis::Graph;
use rayon::slice::ParallelSliceMut;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use crate::{CooMatrix, CsrMatrix, Vector};

/// A non-overlapping partition of square matrix (or 'graph').
pub struct Partition {
    pub mat: Arc<CsrMatrix>,
    pub node_to_agg: Vec<usize>,
    pub agg_to_node: Vec<BTreeSet<usize>>,
}

/// Use Metis C API bindings to partition a symmetric matrix (undirected graph). Here we min-max map the modularity
/// weights into quantized integer values as required by Metis. Do not try to partition large
/// graphs into many (thousdands) of parts or Metis will crash. Either apply this recursively with
/// some approach or use the greedy matching algorithm. Also, if your matrix has more than
/// `i32::MAX` non-zeros (graph edges) Metis will crash.
pub fn metis_n(near_null: &'_ Vector, mat: Arc<CsrMatrix>, n_parts: usize) -> Partition {
    let (a_bar, row_sums, inverse_total) = build_weighted_matrix_coosym(&mat, &near_null);
    let modularity_mat = build_sparse_modularity_matrix(&a_bar, &row_sums, inverse_total);

    let max = modularity_mat
        .data()
        .iter()
        .fold(0.0, |acc, x| if acc > *x { acc } else { *x });
    let min = modularity_mat
        .data()
        .iter()
        .fold(9999999.0, |acc, x| if acc < *x { acc } else { *x });
    let dif = max - min;

    let xadj: Vec<i32> = modularity_mat
        .indptr()
        .as_slice()
        .unwrap()
        .iter()
        .map(|i| *i as i32)
        .collect();
    let adjncy: Vec<i32> = modularity_mat.indices().iter().map(|j| *j as i32).collect();

    let weigts: Vec<i32> = modularity_mat
        .data()
        .iter()
        .map(|val| (1e6_f64 * (val - min) / dif).ceil() as i32 + 1)
        .collect();

    let graph = Graph::new(1_i32, n_parts as i32, &xadj, &adjncy).unwrap();
    let graph = graph.set_adjwgt(&weigts);
    let mut partition = vec![0_i32; modularity_mat.rows()];
    graph.part_kway(&mut partition).unwrap();

    let mut node_to_agg = Vec::with_capacity(mat.rows());
    let mut agg_to_node = vec![BTreeSet::new(); n_parts];

    for (i, agg) in partition.into_iter().map(|agg| agg as usize).enumerate() {
        node_to_agg.push(agg);
        agg_to_node[agg].insert(i);
    }

    #[cfg(debug_assertions)]
    {
        let all: BTreeSet<usize> = (0..mat.rows()).collect();
        let mut running = BTreeSet::new();
        for agg in agg_to_node.iter() {
            assert!(running.is_disjoint(agg));
            running.extend(agg);
        }
        assert!(running.is_subset(&all));
        assert!(running.symmetric_difference(&all).next().is_none());
    }

    Partition {
        mat,
        node_to_agg,
        agg_to_node,
    }
}

pub fn build_sparse_modularity_matrix(
    a_bar: &Vec<(usize, usize, f64)>,
    row_sums: &Vector,
    inverse_total: f64,
) -> CsrMatrix {
    let mut mod_mat = CooMatrix::new((row_sums.len(), row_sums.len()));
    a_bar.iter().copied().for_each(|(i, j, w)| {
        if w > 0.0 {
            let modularity_ij = w - inverse_total * row_sums[i] * row_sums[j];
            if modularity_ij > 0.0 {
                mod_mat.add_triplet(i, j, modularity_ij);
                mod_mat.add_triplet(j, i, modularity_ij);
            }
        }
    });
    mod_mat.to_csr()
}

/// Partition a symmetric matrix (undirected graph) with greedy matching of modularity weights.
/// Performs pairwise matching until the desired coarsening factor is reached:
///
/// `fine_size / coarse_size > coarsening_factor`
///
/// To achieve balance aggregates, choose a `max_agg_size = ceil(coarsening_factor)` or `ceil(coarsening_factor) + 1`.
pub fn modularity_matching_partition(
    mat: Arc<CsrMatrix>,
    near_null: &Vector,
    coarsening_factor: f64,
    max_agg_size: Option<usize>,
) -> Partition {
    let ndofs = mat.rows();
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

    #[cfg(debug_assertions)]
    {
        let all: BTreeSet<usize> = (0..ndofs).collect();
        let mut running = BTreeSet::new();
        for agg in aggs.iter() {
            assert!(running.is_disjoint(agg));
            running.extend(agg);
        }
        assert!(running.is_subset(&all));
        assert!(running.symmetric_difference(&all).next().is_none());
    }

    Partition {
        mat,
        node_to_agg,
        agg_to_node: aggs,
    }
}

fn build_weighted_matrix_coosym(
    mat: &CsrMatrix,
    near_null: &Vector,
) -> (Vec<(usize, usize, f64)>, Vector, f64) {
    let mut row_sums: Vector = Vector::from_elem(mat.rows(), 0.0);
    let mut total: f64 = 0.0;
    let mut mat_bar = Vec::new();

    for (i, row) in mat.outer_iterator().enumerate() {
        for (j, val) in row.iter().filter(|(j, _)| i < *j) {
            let strength_ij = -val * near_null[i] * near_null[j];
            mat_bar.push((i, j, strength_ij));
            row_sums[i] += strength_ij;
            row_sums[j] += strength_ij;
            total += 2.0 * strength_ij;
        }
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
            row_sums.len(),
            total_bad / (counter as f64),
            min
        );
    }

    (mat_bar, row_sums, 1.0 / (total - total_bad))
}

fn greedy_matching(
    a_bar: &mut Vec<(usize, usize, f64)>,
    row_sums: &mut Vector,
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
    let mut new_row_sums = Vector::from_elem(new_agg_count, 0.0);

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
    #[cfg(debug_assertions)]
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

/// Partition a symmetric matrix (undirected graph) with greedy locally optimal aggregation.
/// This particular variant utilizes the notion of coarse and fine (C/F) interpolation sets and selects
/// coarse points as the max near_null[i] within an aggregate. Used for interpolation schemes which
/// require C/F partitioning of the matrix.
///
/// Continues aggregation until the desired coarsening factor is reached:
///
/// `fine_size / coarse_size > coarsening_factor`
pub fn cf_aggregation(
    mat: Arc<CsrMatrix>,
    near_null: &Vector,
    coarsening_factor: f64,
) -> (Partition, Vec<usize>) {
    let ndofs = mat.rows();
    let fine_ndofs = ndofs as f64;
    let mut aggs = (0..ndofs)
        .map(|i| {
            let mut agg = BTreeSet::new();
            agg.insert(i);
            agg
        })
        .collect();

    let (mut a_bar, _row_sums, _inverse_total) = build_weighted_matrix_coosym(&mat, &near_null);

    a_bar = a_bar.into_iter().filter(|(_i, _j, w)| *w > 0.0).collect();

    let mut current_near_null = near_null.clone();
    let mut coarse_indices: Vec<usize> = (0..ndofs).collect();
    loop {
        let max_w = greedy_aggregation(&mut current_near_null, &mut a_bar, &mut aggs);
        coarse_indices = max_w
            .into_iter()
            .map(|max_idx| coarse_indices[max_idx])
            .collect();
        let coarse_ndofs = current_near_null.len() as f64;
        let current_cf = fine_ndofs / coarse_ndofs;

        if a_bar.is_empty() {
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

    (
        Partition {
            mat,
            node_to_agg,
            agg_to_node: aggs,
        },
        coarse_indices,
    )
}

fn greedy_aggregation(
    near_null: &mut Vector,
    a_bar: &mut Vec<(usize, usize, f64)>,
    aggs: &mut Vec<BTreeSet<usize>>,
) -> Vec<usize> {
    let vertex_count = near_null.len();

    let mut row_sums = Vector::from_elem(near_null.len(), 0.0);
    let mut total = 0.0f64;
    for (i, j, v) in a_bar.iter() {
        row_sums[*i] += v;
        row_sums[*j] += v;
        total += v;
    }

    let mut neg_count = 0;
    let mut neq_total = 0.0;
    for sum in row_sums.iter_mut() {
        if *sum < 0.0 {
            neg_count += 1;
            neq_total += *sum;
            *sum = 0.0;
        }
    }
    if neg_count > 0 {
        warn!(
            "{} rows had negative rowsums with total {:.2e}",
            neg_count, neq_total
        );
    }
    let inverse_total = 1.0 / total;

    let mut wants_to_merge: Vec<(usize, usize, f64)> = a_bar
        .iter()
        .copied()
        .map(|(i, j, w)| {
            let modularity_ij = w - inverse_total * row_sums[i] * row_sums[j];
            (i, j, modularity_ij)
        })
        .collect();
    wants_to_merge.par_sort_by(|(_, _, w1), (_, _, w2)| w1.partial_cmp(w2).unwrap());

    let mut alive: HashSet<usize> = (0..vertex_count).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(vertex_count / 2);
    let mut local_aggs = vec![None; vertex_count];

    // First find all locally optimal matches
    for (i, j, _w) in wants_to_merge.iter() {
        //for (i, j, _w) in a_bar.iter() {
        if alive.contains(i) && alive.contains(j) {
            alive.remove(i);
            alive.remove(j);
            let agg_id = pairs.len();
            local_aggs[*i] = Some(agg_id);
            local_aggs[*j] = Some(agg_id);
            pairs.push((*i, *j));
        }
    }

    assert!(!pairs.is_empty());

    // Then any unmatched vertex will join the pair it likes most
    for (i, j, _w) in wants_to_merge.iter() {
        //for (i, j, _w) in a_bar.iter() {
        if alive.contains(i) {
            local_aggs[*i] = local_aggs[*j];
            alive.remove(i);
        } else if alive.contains(j) {
            local_aggs[*j] = local_aggs[*i];
            alive.remove(j);
        }
    }

    let local_aggs: Vec<usize> = local_aggs
        .into_iter()
        .map(|option_agg| option_agg.expect("All vertices must have an aggregate..."))
        .collect();

    // Update the aggregates and create intermediate P
    let new_agg_count = pairs.len();
    let mut max_w: Vec<Option<usize>> = vec![None; new_agg_count];
    let mut new_aggs = vec![BTreeSet::new(); new_agg_count];
    for (i, new_agg_id) in local_aggs.iter().enumerate() {
        new_aggs[*new_agg_id].extend(&aggs[i]);

        match max_w[*new_agg_id] {
            None => max_w[*new_agg_id] = Some(i),
            Some(j) => {
                if near_null[i].abs() > near_null[j].abs() {
                    max_w[*new_agg_id] = Some(i);
                }
            }
        }
    }
    *aggs = new_aggs;
    let max_w: Vec<usize> = max_w
        .into_iter()
        .map(|option_i| option_i.expect("All aggregates should have a max..."))
        .collect();

    // Special P^T A_bar P
    *a_bar = a_bar
        .iter()
        .cloned()
        .map(|(i, j, v_ij)| {
            let ic = local_aggs[i];
            let jc = local_aggs[j];
            let pt_ic_i = near_null[i] / near_null[max_w[ic]];
            let p_j_jc = near_null[j] / near_null[max_w[jc]];
            let v_ic_jc = pt_ic_i * v_ij * p_j_jc;
            (ic, jc, v_ic_jc)
        })
        .filter(|(ic, jc, _)| *ic != *jc)
        .map(|(ic, jc, v_ic_jc)| {
            if ic > jc {
                (jc, ic, v_ic_jc)
            } else {
                (ic, jc, v_ic_jc)
            }
        })
        .collect();

    a_bar.sort_by(|a, b| {
        if a.0 != b.0 {
            a.0.cmp(&b.0)
        } else {
            a.1.cmp(&b.1)
        }
    });
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

    a_bar.retain(|(_, _, v)| *v > 0.0);

    *near_null = Vector::from_iter(max_w.iter().map(|i| near_null[*i]));
    //let agg_sizes_str: String = aggs.iter().map(|agg| format!("{:3}", agg.len())).collect();
    //trace!("agg sizes:\n{}", agg_sizes_str);

    max_w
}
