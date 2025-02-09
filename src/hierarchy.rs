use core::fmt;

use std::sync::Arc;

use ndarray_linalg::Norm;
use sprs::is_symmetric;

use crate::interpolation::{classical, smoothed_aggregation, InterpolationType};
use crate::partitioner::{modularity_matching_partition, Partition};
use crate::{CsrMatrix, Vector};

#[derive(Clone)]
pub struct Hierarchy {
    mat: Arc<CsrMatrix>,
    restrictions: Vec<Arc<CsrMatrix>>,
    interpolations: Vec<Arc<CsrMatrix>>,
    coarse_mats: Vec<Arc<CsrMatrix>>,
    partitions: Vec<Arc<Partition>>,
    near_nulls: Vec<Arc<Vector>>,
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
        }
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
    ) -> Vector {
        let fine_mat = self
            .get_coarse_mats()
            .last()
            .map_or(self.get_mat(0), |m| m.clone());

        let partition = modularity_matching_partition(
            fine_mat.clone(),
            near_null,
            coarsening_factor,
            Some(coarsening_factor.ceil() as usize),
        );

        /*
        let (partition, _coarse_indices) =
            cf_aggregation(fine_mat.clone(), near_null, coarsening_factor);
        */

        let partition = Arc::new(partition);
        let mut normalized_nn = near_null.clone();
        normalized_nn /= near_null.norm();
        self.near_nulls.push(Arc::new(normalized_nn));
        self.partitions.push(partition.clone());

        let (coarse_near_null, r, p, mut mat_coarse) = match interpolation_type {
            InterpolationType::Classical => classical(&fine_mat, &partition, near_null),
            InterpolationType::SmoothedAggregation((smoothing_steps, jacobi_weight)) => {
                smoothed_aggregation(
                    &fine_mat,
                    &partition,
                    near_null,
                    smoothing_steps,
                    jacobi_weight,
                )
            }
            InterpolationType::UnsmoothedAggregation => {
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
