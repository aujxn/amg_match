use core::fmt;

use std::sync::Arc;

use ndarray_linalg::Norm;
use sprs::is_symmetric;

use crate::interpolation::{
    classical, smoothed_aggregation, smoothed_aggregation2, InterpolationInfo, InterpolationType,
};
use crate::partitioner::{BlockReductionStrategy, Partition, PartitionBuilder};
use crate::{CsrMatrix, Matrix, Vector};

#[derive(Clone)]
pub struct Hierarchy {
    mat: Arc<CsrMatrix>,
    restrictions: Vec<Arc<CsrMatrix>>,
    interpolations: Vec<Arc<CsrMatrix>>,
    coarse_mats: Vec<Arc<CsrMatrix>>,
    partitions: Vec<Arc<Partition>>,
    near_nulls: Vec<Arc<Vector>>,
    pub vdims: Vec<usize>,
}

impl fmt::Debug for Hierarchy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fine_size = self.mat.rows();
        let mut sizes: Vec<usize> = vec![fine_size];
        sizes.extend(self.coarse_mats.iter().map(|a| a.rows()));

        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.coarse_mats.iter().map(|a| a.nnz()));

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
    pub fn new(mat: Arc<CsrMatrix>) -> Self {
        Self {
            mat,
            restrictions: Vec::new(),
            interpolations: Vec::new(),
            coarse_mats: Vec::new(),
            partitions: Vec::new(),
            near_nulls: Vec::new(),
            vdims: Vec::new(),
        }
    }

    /*
    pub fn consolidate(&mut self, cf: f64) {
        let mut new_restrictions = Vec::new();
        let mut new_interpolations = Vec::new();
        let mut new_coarse_mats = Vec::new();
        let mut new_near_nulls = Vec::new();
        let mut new_vdims = Vec::new();
        let new_partitions = Vec::new();

        let mut old_level = 0;
        let mut fine_nnz = self.mat.nnz() as f64;
        while old_level < self.coarse_mats.len() {
            new_near_nulls.push(self.near_nulls[old_level].clone());
            new_vdims.push(self.vdims[old_level]);
            let mut new_p = self.interpolations[old_level].clone();
            //let mut current_cf = new_p.rows() as f64 / new_p.cols() as f64;
            let mut current_cf = fine_nnz / self.coarse_mats[old_level].nnz() as f64;
            old_level += 1;
            //while current_cf < cf && old_level < self.coarse_mats.len() {
            while current_cf < 1.5 && old_level < self.coarse_mats.len() {
                new_p = Arc::new(&*new_p * &*self.interpolations[old_level]);
                //current_cf = new_p.rows() as f64 / new_p.cols() as f64;
                current_cf = fine_nnz / self.coarse_mats[old_level].nnz() as f64;
                old_level += 1;
            }
            let new_p = new_p.to_csr();
            let new_r = new_p.transpose_view().to_csr();
            new_interpolations.push(Arc::new(new_p));
            new_restrictions.push(Arc::new(new_r));
            new_coarse_mats.push(self.coarse_mats[old_level - 1].clone());
        }
        self.interpolations = new_interpolations;
        self.restrictions = new_restrictions;
        self.coarse_mats = new_coarse_mats;
        self.partitions = new_partitions;
        self.vdims = new_vdims;
        self.near_nulls = new_near_nulls;
    }
    */

    pub fn push_level(
        &mut self,
        mat: Arc<CsrMatrix>,
        r: Arc<CsrMatrix>,
        p: Arc<CsrMatrix>,
        partition: Arc<Partition>,
        near_null: Arc<Vector>,
        vdim: usize,
    ) {
        self.coarse_mats.push(mat);
        self.restrictions.push(r);
        self.interpolations.push(p);
        self.partitions.push(partition);
        self.near_nulls.push(near_null);
        self.vdims.push(vdim);
    }

    pub fn from_partitions(
        fine_mat: Arc<CsrMatrix>,
        partitions: Vec<Arc<Partition>>,
        near_null: &Matrix,
    ) -> Self {
        let mut mat = (*fine_mat).clone();
        let mut restrictions = Vec::new();
        let mut interpolations = Vec::new();
        let mut coarse_mats = Vec::new();
        let mut near_nulls = Vec::new();
        let mut vdims = Vec::new();
        let mut next_near_null = near_null.clone();
        let mut block_size = 1;
        vdims.push(block_size);

        let mut collapsed = Vector::zeros(next_near_null.nrows());
        for near_null in next_near_null.columns().into_iter() {
            collapsed = collapsed + near_null;
        }
        near_nulls.push(Arc::new(collapsed));

        for partition in partitions.iter() {
            let (coarse_near_null, r, p, mat_coarse) =
                smoothed_aggregation2(&mat, &**partition, block_size, &next_near_null);
            mat = mat_coarse.clone();
            coarse_mats.push(Arc::new(mat_coarse));
            interpolations.push(Arc::new(p));
            restrictions.push(Arc::new(r));
            next_near_null = coarse_near_null.clone();
            block_size = next_near_null.ncols();
            //near_nulls.push(Arc::new(next_near_null.column(0).to_owned()));
            let mut collapsed = Vector::zeros(next_near_null.nrows());
            for near_null in next_near_null.columns().into_iter() {
                collapsed = collapsed + near_null;
            }
            near_nulls.push(Arc::new(collapsed));
            vdims.push(block_size);
        }

        Self {
            mat: fine_mat,
            restrictions,
            interpolations,
            coarse_mats,
            partitions,
            near_nulls,
            vdims,
        }
    }

    pub fn print_table(&self) {
        let infos = self
            .interpolations
            .iter()
            .map(|p| InterpolationInfo::new(p))
            .collect();
        InterpolationInfo::display(&infos);
    }

    /// Total nnz / finest level nnz
    pub fn op_complexity(&self) -> f64 {
        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.coarse_mats.iter().map(|a| a.nnz()));
        let total_nnz_coarse = nnzs.iter().sum::<usize>() as f64;
        total_nnz_coarse / (fine_nnz as f64)
    }

    /// Number of levels in the hierarchy.
    pub fn levels(&self) -> usize {
        self.coarse_mats.len() + 1
    }

    /// Check if the hierarchy has any levels
    pub fn is_empty(&self) -> bool {
        self.coarse_mats.is_empty()
    }

    /// Adds a level to the hierarchy given a partitioning of the matrix graph, a near-null vector,
    /// and interpolation method
    pub fn add_level(
        &mut self,
        near_null: &Vector,
        coarsening_factor: f64,
        interpolation_type: InterpolationType,
        block_size: usize,
    ) -> Vector {
        let fine_mat = self
            .get_coarse_mats()
            .last()
            .map_or(self.get_mat(0), |m| m.clone());

        let mut normalized_nn = near_null.clone();
        normalized_nn /= near_null.norm();
        self.near_nulls.push(Arc::new(normalized_nn));

        let mut builder = PartitionBuilder::new(fine_mat.clone(), Arc::new(near_null.clone()));
        builder.coarsening_factor = coarsening_factor;
        builder.max_agg_size = Some(coarsening_factor.ceil() as usize);
        if block_size > 1 {
            builder.block_reduction_strategy = Some(BlockReductionStrategy::default());
            builder.vector_dim = block_size;
        }

        let (coarse_near_null, r, p, mut mat_coarse) = match interpolation_type {
            InterpolationType::Classical => classical(&fine_mat, near_null),
            InterpolationType::SmoothedAggregation((smoothing_steps, jacobi_weight)) => {
                let partition = builder.build();

                /*
                let mut max_w: Vec<Option<usize>> = vec![None; partition.agg_to_node.len()];
                for (fine_idx, coarse_idx) in partition.node_to_agg.iter().enumerate() {
                    match max_w[*coarse_idx] {
                        None => max_w[*coarse_idx] = Some(fine_idx),
                        Some(other_fine_idx) => {
                            if near_null[fine_idx].abs() > near_null[other_fine_idx].abs() {
                                max_w[*coarse_idx] = Some(fine_idx);
                            }
                        }
                    }
                }
                let coarse_indices: Vec<usize> = max_w
                    .into_iter()
                    .map(|option_i| option_i.expect("All aggregates should have a max..."))
                    .collect();
                */

                /*
                let (partition, coarse_indices) =
                    cf_aggregation(fine_mat.clone(), near_null, coarsening_factor);
                */
                let partition = Arc::new(partition);
                self.partitions.push(partition.clone());
                smoothed_aggregation(
                    &fine_mat,
                    &partition,
                    near_null,
                    smoothing_steps,
                    jacobi_weight,
                )
                //smoothed_aggregation2(&fine_mat, &partition, near_null, &coarse_indices)
            }
            InterpolationType::UnsmoothedAggregation => {
                let partition = builder.build();

                /*
                let (partition, _coarse_indices) =
                    cf_aggregation(fine_mat.clone(), near_null, coarsening_factor);
                */

                let partition = Arc::new(partition);
                self.partitions.push(partition.clone());
                smoothed_aggregation(&fine_mat, &partition, near_null, 0, 0.0)
            }
        };

        trace!(
            "added level: {}. num vertices coarse: {} nnz: {}",
            self.levels() + 1,
            p.cols(),
            mat_coarse.nnz()
        );

        // TODO maybe not needed / helpful and this implementation is lazy and inefficient
        if !is_symmetric(&mat_coarse) {
            mat_coarse = &mat_coarse.view() + &mat_coarse.transpose_view();
            mat_coarse.map_inplace(|v| v * 0.5);
        }

        self.coarse_mats.push(Arc::new(mat_coarse));
        self.interpolations.push(Arc::new(p));
        self.restrictions.push(Arc::new(r));
        self.vdims.push(block_size);

        coarse_near_null
    }

    /// Get a single P matrix from the hierarchy.
    pub fn get_restriction(&self, level: usize) -> &Arc<CsrMatrix> {
        &self.restrictions[level]
    }

    /// Get a single P^T matrix from the hierarchy.
    pub fn get_interpolation(&self, level: usize) -> &Arc<CsrMatrix> {
        &self.interpolations[level]
    }

    pub fn get_near_null(&self, level: usize) -> &Arc<Vector> {
        &self.near_nulls[level]
    }

    /// Get a reference to the matrices Vec.
    pub fn get_coarse_mats(&self) -> &[Arc<CsrMatrix>] {
        &self.coarse_mats
    }

    /// Get a reference to the P matrices Vec.
    pub fn get_restrictions(&self) -> &Vec<Arc<CsrMatrix>> {
        &self.restrictions
    }

    /// Get a reference to the P^T matrices Vec.
    pub fn get_interpolations(&self) -> &Vec<Arc<CsrMatrix>> {
        &self.interpolations
    }

    /// Get a reference to the partitions Vec.
    pub fn get_partitions(&self) -> &Vec<Arc<Partition>> {
        &self.partitions
    }

    pub fn get_near_nulls(&self) -> &Vec<Arc<Vector>> {
        &self.near_nulls
    }

    pub fn get_mat(&self, level: usize) -> Arc<CsrMatrix> {
        if level == 0 {
            self.mat.clone()
        } else {
            self.coarse_mats[level - 1].clone()
        }
    }

    pub fn set_fine_mat(&mut self, fine_mat: Arc<CsrMatrix>) {
        self.mat = fine_mat.clone();
        self.coarse_mats = Vec::new();

        let mat = &*fine_mat;
        for (pt, p) in self
            .restrictions
            .iter()
            .cloned()
            .zip(self.interpolations.iter().cloned())
        {
            let pt = &*pt;
            let p = &*p;
            let coarse: CsrMatrix;
            if let Some(prev_mat) = self.coarse_mats.last() {
                let prev_mat = &*(prev_mat.clone());
                coarse = (pt * &(prev_mat * p)).to_csr();
            } else {
                coarse = (pt * &(mat * p)).to_csr();
            }
            self.coarse_mats.push(Arc::new(coarse));
        }
    }

    pub fn get_nnzs(&self) -> Vec<usize> {
        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.coarse_mats.iter().map(|a| a.nnz()));
        nnzs
    }

    pub fn get_dims(&self) -> Vec<usize> {
        let fine_size = self.mat.rows();
        let mut sizes: Vec<usize> = vec![fine_size];
        sizes.extend(self.coarse_mats.iter().map(|a| a.rows()));
        sizes
    }

    pub fn memory_complexity(&self) -> f64 {
        let nnzs = self.get_nnzs();
        let fine_nnz = self.mat.nnz();
        let total_nnz = nnzs.iter().sum::<usize>() as f64;
        total_nnz / (fine_nnz as f64)
    }
}
