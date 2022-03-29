use indexmap::IndexSet;
use rand::prelude::*;
use sprs::{CsMat, TriMat};

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
