use core::fmt;
use nalgebra::base::DVector;
use nalgebra_sparse::csr::CsrMatrix;

use std::sync::Arc;

use crate::interpolation::{classical, smoothed_aggregation, InterpolationType};
use crate::partitioner::{modularity_matching_partition, Partition};

#[derive(Clone)]
pub struct Hierarchy {
    mat: Arc<CsrMatrix<f64>>,
    restrictions: Vec<Arc<CsrMatrix<f64>>>,
    interpolations: Vec<Arc<CsrMatrix<f64>>>,
    coarse_mats: Vec<Arc<CsrMatrix<f64>>>,
    partitions: Vec<Arc<Partition>>,
    near_nulls: Vec<Arc<DVector<f64>>>,
}

impl fmt::Debug for Hierarchy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fine_size = self.mat.nrows();
        let mut sizes: Vec<usize> = vec![fine_size];
        sizes.extend(self.coarse_mats.iter().map(|a| a.nrows()));

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
    pub fn new(mat: Arc<CsrMatrix<f64>>) -> Self {
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
        near_null: &DVector<f64>,
        coarsening_factor: f64,
        interpolation_type: InterpolationType,
    ) -> DVector<f64> {
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

        let partition = Arc::new(partition);
        self.near_nulls.push(Arc::new(near_null.clone()));
        self.partitions.push(partition.clone());

        let (coarse_near_null, r, p, mat_coarse) = match interpolation_type {
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
            p.ncols(),
            mat_coarse.nnz()
        );

        self.coarse_mats.push(Arc::new(mat_coarse));
        self.interpolations.push(Arc::new(p));
        self.restrictions.push(Arc::new(r));

        coarse_near_null
    }

    /// Get a single P matrix from the hierarchy.
    pub fn get_restriction(&self, level: usize) -> &Arc<CsrMatrix<f64>> {
        &self.restrictions[level]
    }

    /// Get a single P^T matrix from the hierarchy.
    pub fn get_interpolation(&self, level: usize) -> &Arc<CsrMatrix<f64>> {
        &self.interpolations[level]
    }

    pub fn get_near_null(&self, level: usize) -> &Arc<DVector<f64>> {
        &self.near_nulls[level]
    }

    /// Get a reference to the matrices Vec.
    pub fn get_coarse_mats(&self) -> &[Arc<CsrMatrix<f64>>] {
        &self.coarse_mats
    }

    /// Get a reference to the P matrices Vec.
    pub fn get_restrictions(&self) -> &Vec<Arc<CsrMatrix<f64>>> {
        &self.restrictions
    }

    /// Get a reference to the P^T matrices Vec.
    pub fn get_interpolations(&self) -> &Vec<Arc<CsrMatrix<f64>>> {
        &self.interpolations
    }

    pub fn get_near_nulls(&self) -> &Vec<Arc<DVector<f64>>> {
        &self.near_nulls
    }

    pub fn get_mat(&self, level: usize) -> Arc<CsrMatrix<f64>> {
        if level == 0 {
            self.mat.clone()
        } else {
            self.coarse_mats[level - 1].clone()
        }
    }

    pub fn get_nnzs(&self) -> Vec<usize> {
        let fine_nnz = self.mat.nnz();
        let mut nnzs: Vec<usize> = vec![fine_nnz];
        nnzs.extend(self.coarse_mats.iter().map(|a| a.nnz()));
        nnzs
    }

    pub fn get_dims(&self) -> Vec<usize> {
        let fine_size = self.mat.nrows();
        let mut sizes: Vec<usize> = vec![fine_size];
        sizes.extend(self.coarse_mats.iter().map(|a| a.nrows()));
        sizes
    }

    pub fn memory_complexity(&self) -> f64 {
        let nnzs = self.get_nnzs();
        let fine_nnz = self.mat.nnz();
        let total_nnz = nnzs.iter().sum::<usize>() as f64;
        total_nnz / (fine_nnz as f64)
    }
}
