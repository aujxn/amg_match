//! This module contains methods that partitions the matrix into hierarchies.
//! This could potentially be moved to a seperate crate, since I have copied
//! this code into other projects as well.

use core::fmt;
use indexmap::IndexSet;
use metis::Graph;
use nalgebra::base::DVector;
//use nalgebra::{DMatrix, QR};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rayon::slice::ParallelSliceMut;
use std::borrow::Borrow;
use std::collections::VecDeque;

use std::sync::Arc;
//use crate::parallel_ops::spmm;

#[derive(Copy, Clone, Debug)]
pub enum InterpolationType {
    UnsmoothedAggregation,
    SmoothedAggregation(usize),
    Classical,
}

//TODO bring back tests you deleted when preconditioner refactor happened

/// Resulting object from running the modularity matching algorithm.
///
#[derive(Clone)]
pub struct Hierarchy {
    mat: Arc<CsrMatrix<f64>>,
    // TODO Arc everything because duplication bad
    partition_matrices: Vec<CsrMatrix<f64>>,
    interpolation_matrices: Vec<CsrMatrix<f64>>,
    pub matrices: Vec<Arc<CsrMatrix<f64>>>,
    pub restriction_matrices: Vec<CsrMatrix<f64>>,
    pub near_nulls: Vec<DVector<f64>>,
}

impl fmt::Debug for Hierarchy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fine_size = self.mat.nrows();
        let mut sizes: Vec<usize> = vec![fine_size];
        sizes.extend(self.matrices.iter().map(|a| a.nrows()));

        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.matrices.iter().map(|a| a.nnz()));

        let total_nnz_coarse = nnzs.iter().sum::<usize>() as f32;
        let complexity = self.op_complexity();
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
    pub fn new(mat: Arc<CsrMatrix<f64>>) -> Self {
        Self {
            mat,
            partition_matrices: Vec::new(),
            interpolation_matrices: Vec::new(),
            restriction_matrices: Vec::new(),
            matrices: Vec::new(),
            near_nulls: Vec::new(),
        }
    }

    pub fn consolidate(&mut self, target_cf: f64) {
        let mut new_partitions = Vec::new();
        let mut new_interpolations = Vec::new();
        let mut new_matrices = Vec::new();
        let mut base = 0;
        while base < self.partition_matrices.len() {
            let mut new_p = self.partition_matrices[base].clone();
            let size = new_p.nrows() as f64;
            let mut i = base + 1;
            let mut cf = size / (new_p.ncols() as f64);
            while cf < target_cf && i < self.partition_matrices.len() {
                new_p = new_p * &self.partition_matrices[i];
                cf = size / (new_p.ncols() as f64);
                i += 1;
            }
            let pt = new_p.transpose();
            if new_matrices.is_empty() {
                let fine_mat: &CsrMatrix<f64> = self.mat.borrow();
                let coarse_mat = &pt * &(fine_mat * &new_p);
                new_matrices.push(Arc::new(coarse_mat));
            } else {
                let fine_mat: &CsrMatrix<f64> = new_matrices.last().unwrap().borrow();
                let coarse_mat: CsrMatrix<f64> = &pt * &(fine_mat * &new_p);
                new_matrices.push(Arc::new(coarse_mat));
            }
            new_partitions.push(new_p);
            new_interpolations.push(pt);
            base = i;
        }
        self.partition_matrices = new_partitions;
        self.interpolation_matrices = new_interpolations;
        self.matrices = new_matrices;
    }

    pub fn from_hierarchy(
        mat: Arc<CsrMatrix<f64>>,
        partition_matrices: Vec<CsrMatrix<f64>>,
    ) -> Self {
        let interpolation_matrices: Vec<CsrMatrix<f64>> =
            partition_matrices.iter().map(|p| p.transpose()).collect();
        let mut matrices: Vec<Arc<CsrMatrix<f64>>> = Vec::new();

        for (p, p_t) in partition_matrices.iter().zip(interpolation_matrices.iter()) {
            if let Some(mat) = matrices.last() {
                let rc: Arc<CsrMatrix<f64>> = mat.clone();
                let prev: &CsrMatrix<f64> = rc.borrow();
                let coarse_mat = p_t * &(prev * p);
                matrices.push(Arc::new(coarse_mat));
            } else {
                let fine_mat: &CsrMatrix<f64> = mat.borrow();
                let coarse_mat = p_t * &(fine_mat * p);
                matrices.push(Arc::new(coarse_mat));
            }
        }

        // TODO should be an argument to this? or construct from P?
        let restriction_matrices = Vec::new();
        let near_nulls = Vec::new();

        Self {
            mat,
            partition_matrices,
            interpolation_matrices,
            near_nulls,
            restriction_matrices,
            matrices,
        }
    }

    pub fn op_complexity(&self) -> f32 {
        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.matrices.iter().map(|a| a.nnz()));
        let total_nnz_coarse = nnzs.iter().sum::<usize>() as f32;
        total_nnz_coarse / (fine_nnz as f32)
    }

    /// Number of levels in the hierarchy.
    pub fn levels(&self) -> usize {
        self.matrices.len() + 1
    }

    /// Check if the hierarchy has any levels
    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    /// Adds a level to the hierarchy given a partitioning of the matrix graph, a near-null vector,
    /// and interpolation method
    pub fn add_level(
        &mut self,
        partition_mat: CsrMatrix<f64>,
        near_null: &DVector<f64>,
        interpolation_type: InterpolationType,
    ) -> DVector<f64> {
        let coarse_near_null = match interpolation_type {
            InterpolationType::Classical => {
                self.add_classical_interpolant(partition_mat, near_null)
            }
            InterpolationType::SmoothedAggregation(smoothing_steps) => {
                self.smoothed_aggregation(partition_mat, near_null, smoothing_steps)
            }
            InterpolationType::UnsmoothedAggregation => {
                self.smoothed_aggregation(partition_mat, near_null, 0)
            }
        };

        trace!(
            "added level: {}. num vertices coarse: {} nnz: {}",
            self.levels(),
            self.get_partitions().last().unwrap().ncols(),
            self.get_matrices().last().unwrap().nnz()
        );
        coarse_near_null
    }

    /*
    pub fn block_smoothed_aggregation(
        &mut self,
        mut partition_mat: CsrMatrix<f64>,
        near_null_space: &DMatrix<f64>,
    ) {
        let fine_mat: &CsrMatrix<f64>;

        if self.matrices.is_empty() {
            fine_mat = self.mat.borrow();
        } else {
            fine_mat = self.matrices.last().unwrap().borrow();
        }

        let n = fine_mat.nrows();
        let n_coarse = partition_mat.ncols();
        self.restriction_matrices.push(partition_mat.transpose());
        let p_transpose = partition_mat.transpose();

        let n_vecs = near_null_space.ncols();
        let mut weighted_interp = CooMatrix::new(n, n_coarse * n_vecs);

        let mut coarse_near_null = DMatrix::zeros(n_coarse * n_vecs, n_vecs);

        for (coarse_i, agg) in p_transpose.row_iter().enumerate() {
            let block = near_null_space.select_rows(agg.col_indices());
            let qr = QR::new(block);
            let (q, r) = qr.unpack();

            for (i, row_data) in agg.col_indices().iter().copied().zip(q.row_iter()) {
                let j = coarse_i * n_vecs;
                weighted_interp.push_matrix(i, j, &row_data);
            }

            for (row_offset, row_data) in r.row_iter().enumerate() {
                coarse_near_null.set_row((coarse_i * n_vecs) + row_offset, &row_data);
            }
        }

        let mut diag_inv = fine_mat.diagonal_as_csr();
        diag_inv
            .values_mut()
            .iter_mut()
            .for_each(|val| *val = 0.66 * val.recip());

        partition_mat = CsrMatrix::from(&weighted_interp);
        partition_mat = &partition_mat - (&diag_inv * (fine_mat * &partition_mat));

        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &partition_mat);
        self.matrices.push(Arc::new(coarse_mat));
        self.partition_matrices.push(partition_mat);
        self.interpolation_matrices.push(p_transpose);
        self.coarse_near_null = Some(coarse_near_null.column(0).into());
    }
    */

    pub fn smoothed_aggregation(
        &mut self,
        mut partition_mat: CsrMatrix<f64>,
        near_null: &DVector<f64>,
        smoothing_steps: usize,
    ) -> DVector<f64> {
        let fine_mat: &CsrMatrix<f64>;

        if self.matrices.is_empty() {
            fine_mat = self.mat.borrow();
        } else {
            fine_mat = self.matrices.last().unwrap().borrow();
        }

        let n_coarse = partition_mat.ncols();
        self.restriction_matrices.push(partition_mat.transpose());
        let p_transpose = partition_mat.transpose();

        let mut coarse_near_null: DVector<f64> = DVector::zeros(n_coarse);
        let mut diag_inv = fine_mat.diagonal_as_csr();
        diag_inv
            .values_mut()
            .iter_mut()
            .for_each(|val| *val = 0.66 * val.recip());

        for (coarse_i, agg) in p_transpose.row_iter().enumerate() {
            let r: f64 = agg
                .col_indices()
                .iter()
                .map(|i| near_null[*i].powf(2.0))
                .sum();
            coarse_near_null[coarse_i] = r.sqrt();
        }

        partition_mat
            .triplet_iter_mut()
            .for_each(|(i, j, w)| *w = near_null[i] / coarse_near_null[j]);

        for _ in 0..smoothing_steps {
            partition_mat = &partition_mat - (&diag_inv * (fine_mat * &partition_mat));
        }

        let p_transpose = partition_mat.transpose();
        let coarse_mat = &p_transpose * &(fine_mat * &partition_mat);
        self.matrices.push(Arc::new(coarse_mat));
        self.partition_matrices.push(partition_mat);
        self.interpolation_matrices.push(p_transpose);
        coarse_near_null
    }

    /// More advanced interpolation than piecewise constants.
    pub fn add_classical_interpolant_v2(
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
        let mut all_coarse = Vec::new();
        let mut strong_neighbors: Vec<IndexSet<usize>> = vec![IndexSet::new(); n];

        // Collect all the strong neighbors. This could be more efficient since
        // we actually only need these for f-vertices and not c-vertices.
        // Later we will also care about the sum of the strengths so we could
        // also calculate that here, the concern is that these strengths may
        // divide by 0 requiring that the set is modified later anyway.
        //
        // Additionally we can add negative strengths to strong neighbors
        // in order to satisfy that the set A_c (i) is not empty but we must
        // be careful that the sum of the strengths doesn't get too small or
        // delta_i blows up. Balancing these competing values seems to be the
        // primary challenge with this approach.
        //
        // Some possible ways to address these challenges:
        // - allow strong neighbors to sometimes come from outside the aggregate
        // - allow distance 2 vertices into the strong neighbor set, somehow requiring a
        // change in the strength calculation
        for (i, row) in fine_mat.row_iter().enumerate() {
            for (j, a_ij) in row.col_indices().iter().zip(row.values().iter()) {
                if i != *j && a_ij * near_null[i] * near_null[*j] > 0.0 {
                    strong_neighbors[i].insert(*j);
                }
            }
        }

        let mut warn_couter = 0;
        let mut coarse_verts = 0;
        for agg in p_transpose.row_iter() {
            let mut coarse_vertices = IndexSet::new();
            let agg_vertices_set: IndexSet<usize> = agg.col_indices().iter().copied().collect();

            for i in agg.col_indices() {
                strong_neighbors[*i] = strong_neighbors[*i]
                    .intersection(&agg_vertices_set)
                    .copied()
                    .collect();
            }

            let mut coarse_neighborhood_set: IndexSet<usize> = IndexSet::new();
            let mut agg_vertices: Vec<(usize, f64)> = agg
                .col_indices()
                .iter()
                .copied()
                .map(|i| (i, near_null[i]))
                .collect();
            agg_vertices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Constructing a cover of the aggregate to be the coarse vertices
            // NOTE: in Panayot's note one of the comments mentions the neighborhood
            // of the aggregate but the algorithm only mentions covering the
            // aggregate itself, should clarify this.
            while !coarse_neighborhood_set.is_superset(&agg_vertices_set) {
                let new_coarse_vertex = loop {
                    // We *want unwrap to crash here* because something is wrong.
                    // The coarse neighborhood isn't covering the entire aggregate
                    // and unwrap implies that the aggregate ran out of vertices...
                    let coarse_vertex_candidate = agg_vertices.pop().unwrap().0;
                    if !coarse_vertices.contains(&coarse_vertex_candidate) {
                        break coarse_vertex_candidate;
                    }
                };
                coarse_neighborhood_set.extend(
                    fine_mat
                        .get_row(new_coarse_vertex)
                        .unwrap()
                        .col_indices()
                        .iter(),
                );
                coarse_vertices.insert(new_coarse_vertex);
                coarse_verts += 1;
            }

            let f_vertices: IndexSet<usize> = agg_vertices_set
                .difference(&coarse_vertices)
                .copied()
                .collect();
            for i in f_vertices {
                if strong_neighbors[i].is_disjoint(&coarse_vertices) {
                    warn!(
                        "Vertex {} with weight {:.2e} has no coarse vertices to interpolate from.",
                        i, near_null[i]
                    );

                    warn_couter += 1;

                    // At this point we probably need to decide between adding something to the
                    // strong set versus adding something to the coarse vertices.

                    /*
                    let mut neighborhood_i: IndexSet<usize> =
                        fine_mat.row(*i).col_indices().iter().copied().collect();
                    neighborhood_i = neighborhood_i
                        .intersection(&coarse_vertices)
                        .copied()
                        .collect();
                    let strong_addition = None;
                    let weight = f64::MIN;

                    for coarse_index in neighborhood_i {
                        let new_weight =
                    }
                    */
                }
            }

            all_coarse.push(coarse_vertices);
        }
        panic!(
            "warn counter: {} fine verts: {} coarse verts: {}",
            warn_couter, n, coarse_verts
        );
    }

    /// More advanced interpolation than piecewise constants.
    pub fn add_classical_interpolant(
        &mut self,
        partition_mat: CsrMatrix<f64>,
        near_null: &DVector<f64>,
    ) -> DVector<f64> {
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
                        .all(|prod| prod.abs() < 1e-16)
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
                if value.abs() > 10.0 {
                    //warn!("interpolation weight very large: {}", value);
                }
                let jc = all_coarse.get_index_of(ic).unwrap();
                interpolation.push(i, jc, value);
            }
        }

        let interpolation = CsrMatrix::from(&interpolation);
        let coarse_near_null = &new_pt * near_null;
        self.restriction_matrices.push(new_pt);
        let reconstruction = &interpolation * &coarse_near_null;

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
        self.matrices.push(Arc::new(coarse_mat));
        self.partition_matrices.push(interpolation);
        self.interpolation_matrices.push(p_transpose);
        coarse_near_null
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
    pub fn get_matrices(&self) -> &[Arc<CsrMatrix<f64>>] {
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

    pub fn get_mat(&self, level: usize) -> Arc<CsrMatrix<f64>> {
        if level == 0 {
            self.mat.clone()
        } else {
            self.matrices[level - 1].clone()
        }
    }

    pub fn get_nnzs(&self) -> Vec<usize> {
        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.matrices.iter().map(|a| a.nnz()));
        nnzs
    }

    pub fn get_dims(&self) -> Vec<usize> {
        let fine_size = self.mat.nrows();
        let mut sizes: Vec<usize> = vec![fine_size];
        sizes.extend(self.matrices.iter().map(|a| a.nrows()));
        sizes
    }

    pub fn memory_complexity(&self) -> f64 {
        let nnzs = self.get_nnzs();
        let fine_nnz = self.mat.nnz();
        let total_nnz = nnzs.iter().sum::<usize>() as f64;
        total_nnz / (fine_nnz as f64)
    }
}

// TODO aggregate abstraction...
pub fn metis_n(
    near_null: &'_ DVector<f64>,
    mat: &CsrMatrix<f64>,
    n_parts: usize,
) -> CsrMatrix<f64> {
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

    let mut p_coo = CooMatrix::new(n_parts, modularity_mat.nrows());
    for (i, j) in partition.iter().enumerate() {
        p_coo.push(*j as usize, i, 1.0);
    }

    CsrMatrix::from(&p_coo)
}

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

/*
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
*/
