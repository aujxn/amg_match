use indexmap::IndexSet;
use rand::prelude::*;
use sprs::{CsMat, TriMat};

pub struct Hierarchy {
    pub partition_matrices: Vec<CsMat<f64>>,
    pub matrices: Vec<CsMat<f64>>,
}

pub fn modularity(mat: CsMat<f64>) -> Hierarchy {
    let mut level = 0;
    let mut hierarchy = Hierarchy {
        partition_matrices: vec![],
        matrices: vec![mat],
    };

    loop {
        let vertex_count = hierarchy.matrices[level].rows();
        let row_sums: Vec<f64> = hierarchy.matrices[level]
            .outer_iterator()
            .map(|row| row.data().iter().sum())
            .collect();

        let total_sum: f64 = row_sums.iter().sum();
        let inverse_t = 1.0 / total_sum;

        let wants_to_merge: Vec<Option<usize>> = hierarchy.matrices[level]
            .outer_iterator()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .filter(|(j, _)| i != *j)
                    .fold(None, |max, (j, val)| {
                        let delta_q = val - inverse_t * row_sums[i] * row_sums[j];
                        if delta_q < 0.0 {
                            max
                        } else {
                            match max {
                                Some((_, old_delta_q)) => {
                                    if old_delta_q > delta_q {
                                        max
                                    } else {
                                        Some((j, delta_q))
                                    }
                                }
                                None => Some((j, delta_q)),
                            }
                        }
                    })
                    .map(|(j, _)| j)
            })
            .collect();

        if wants_to_merge.iter().all(|x| x.is_none()) {
            return hierarchy;
        }

        let mut alive = vec![true; vertex_count];
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(vertex_count / 2);

        for i in 0..vertex_count {
            let j = wants_to_merge[i];
            if let Some(j) = j {
                if wants_to_merge[j] == Some(i) && alive[j] && alive[i] {
                    pairs.push((i, j));
                    alive[j] = false;
                    alive[i] = false;
                }
            }
        }

        let pairs_count = pairs.len();
        println!("num edges merged: {pairs_count}\n num vertices: {vertex_count}");
        let mut not_merged_vertices: IndexSet<usize> = (0..vertex_count).collect();
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
        let partition_mat = partition_mat.to_csr::<usize>();

        let coarse_mat = &partition_mat.transpose_view().to_owned()
            * &(&hierarchy.matrices[level] * &partition_mat);
        hierarchy.matrices.push(coarse_mat);
        hierarchy.partition_matrices.push(partition_mat);
        level += 1;
    }
}

pub fn lubys(mat: &CsMat<f64>, weights: Option<Vec<f64>>) -> CsMat<f64> {
    let e_ij: Vec<(usize, usize)> = mat
        .iter()
        .filter(|(_, (i, j))| i > j)
        .map(|(_, (i, j))| (i, j))
        .collect();
    let edge_count = e_ij.len();
    let vertex_count = mat.rows();
    println!("num vertices: {}", vertex_count);
    println!("num edges: {}", edge_count);

    let weights = match weights {
        Some(vec) => vec,
        None => {
            let mut rng = rand::thread_rng();
            (0..edge_count).map(|_| rng.gen()).collect()
        }
    };

    let mut edge_vertex = TriMat::new((edge_count, vertex_count));

    for (edge, (i, j)) in e_ij.into_iter().enumerate() {
        edge_vertex.add_triplet(edge, i, 1);
        edge_vertex.add_triplet(edge, j, 1);
    }

    let edge_vertex: CsMat<i32> = edge_vertex.to_csr::<usize>();
    let vertex_edge: CsMat<i32> = edge_vertex.transpose_view().to_owned();
    let edge_edge = &edge_vertex * &vertex_edge;
    println!("dim of edge_edge: {:?}", edge_edge.shape());

    let mut maximal_edges: IndexSet<usize> = IndexSet::with_capacity(edge_count);
    let mut not_maximal_edges: IndexSet<usize> = IndexSet::with_capacity(edge_count);

    for (i, weight) in weights.iter().enumerate() {
        if not_maximal_edges.contains(&i) {
            continue;
        }

        let connected_edges = edge_edge.outer_view(i).expect("out of range in edge_edge");

        if connected_edges
            .iter()
            .filter(|(j, _)| *j != i)
            .all(|(j, _)| *weight > weights[j])
        {
            maximal_edges.insert(i);
            for (j, _) in connected_edges.iter().filter(|(j, _)| *j != i) {
                not_maximal_edges.insert(j);
            }
        }
    }

    let merged_edge_count = maximal_edges.len();
    println!("num edges merged: {}", merged_edge_count);
    let mut not_merged_vertices: IndexSet<usize> = (0..vertex_count).collect();
    let aggregate_count = vertex_count - merged_edge_count;
    let mut partition_mat = TriMat::new((vertex_count, aggregate_count));

    for (aggregate, edge) in maximal_edges.into_iter().enumerate() {
        if let Some(vertex_pair) = edge_vertex.outer_view(edge) {
            let vertex_pair = vertex_pair.indices();
            let i = vertex_pair[0] as usize;
            let j = vertex_pair[1] as usize;
            partition_mat.add_triplet(i, aggregate, 1.0);
            partition_mat.add_triplet(j, aggregate, 1.0);
            assert!(not_merged_vertices.remove(&i));
            assert!(not_merged_vertices.remove(&j));
        } else {
            panic!("row doesn't exist in edge_vertex")
        }
    }

    for (aggregate, vertex) in not_merged_vertices.into_iter().enumerate() {
        partition_mat.add_triplet(vertex, aggregate + merged_edge_count, 1.0);
    }

    return partition_mat.to_csr::<usize>();
}
