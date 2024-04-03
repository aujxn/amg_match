//! Definition of the `LinearOperator` trait as well as implementors
//! of said trait.

use crate::parallel_ops::spmm;
use crate::partitioner::Hierarchy;
use crate::solver::{lsolve, usolve, Direct, Iterative, IterativeMethod, Solver};
use nalgebra::base::DVector;
use nalgebra_sparse::CsrMatrix;
use std::rc::Rc;

pub trait LinearOperator {
    fn apply_mut(&self, vec: &mut DVector<f64>);
    fn apply(&self, vec: &DVector<f64>) -> DVector<f64>;
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>);
}

#[derive(Copy, Clone, Debug)]
pub enum SmootherType {
    L1,
    GaussSeidel,
}

pub struct Identity;
impl Identity {
    pub fn new() -> Self {
        Self
    }
}

impl LinearOperator for Identity {
    fn apply_mut(&self, _vec: &mut DVector<f64>) {}

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        vec.clone()
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
    }
}

pub struct L1 {
    l1_inverse: DVector<f64>,
}

impl LinearOperator for L1 {
    fn apply_mut(&self, vec: &mut DVector<f64>) {
        vec.component_mul_assign(&self.l1_inverse);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        vec.component_mul(&self.l1_inverse)
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(&in_vec.component_mul(&self.l1_inverse))
    }
}

impl L1 {
    pub fn new(mat: &CsrMatrix<f64>) -> Self {
        let diag_sqrt: Vec<f64> = mat
            .diagonal_as_csr()
            .values()
            .iter()
            .map(|a_ii| a_ii.sqrt())
            .collect();
        let diag_sqrt_inv: Vec<f64> = diag_sqrt.iter().map(|val| val.recip()).collect();

        let l1_inverse: Vec<f64> = mat
            .row_iter()
            .enumerate()
            .map(|(i, row_vec)| {
                row_vec
                    .col_indices()
                    .iter()
                    .zip(row_vec.values().iter())
                    .map(|(j, val)| val.abs() * diag_sqrt[i] * diag_sqrt_inv[*j])
                    .sum::<f64>()
                    .recip()
            })
            .collect();
        /*
        let l1_inverse: Vec<f64> = mat
            .row_iter()
            .map(|row_vec| {
                let row_sum_abs: f64 = row_vec.values().iter().map(|val| val.abs()).sum();
                1.0 / row_sum_abs
            })
            .collect();
        */

        let l1_inverse: DVector<f64> = DVector::from(l1_inverse);
        Self { l1_inverse }
    }
}

pub struct ForwardGaussSeidel {
    mat: Rc<CsrMatrix<f64>>,
}

impl LinearOperator for ForwardGaussSeidel {
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        lsolve(&self.mat, out_vec);
    }
    fn apply_mut(&self, r: &mut DVector<f64>) {
        lsolve(&self.mat, r);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out_vec = vec.clone();
        lsolve(&self.mat, &mut out_vec);
        out_vec
    }
}

impl ForwardGaussSeidel {
    pub fn new(mat: Rc<CsrMatrix<f64>>) -> ForwardGaussSeidel {
        Self { mat }
    }
}

pub struct BackwardGaussSeidel {
    mat: Rc<CsrMatrix<f64>>,
}

impl LinearOperator for BackwardGaussSeidel {
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        usolve(&self.mat, out_vec);
    }
    fn apply_mut(&self, r: &mut DVector<f64>) {
        usolve(&self.mat, r);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut out_vec = vec.clone();
        usolve(&self.mat, &mut out_vec);
        out_vec
    }
}

impl BackwardGaussSeidel {
    pub fn new(mat: Rc<CsrMatrix<f64>>) -> BackwardGaussSeidel {
        Self { mat }
    }
}

/*
pub struct SymmetricGaussSeidel {
    diag: DVector<f64>,
    upper_triangle: CsrMatrix<f64>,
    lower_triangle: CsrMatrix<f64>,
}

impl LinearOperator for SymmetricGaussSeidel {
    fn apply(&self, r: &mut DVector<f64>) {
        lsolve(&self.lower_triangle, r);
        r.component_mul_assign(&self.diag);
        usolve(&self.upper_triangle, r);
    }
}

impl SymmetricGaussSeidel {
    pub fn new(mat: &CsrMatrix<f64>) -> SymmetricGaussSeidel {
        let (_, _, diag) = mat.diagonal_as_csr().disassemble();
        let diag = DVector::from(diag);

        let lower_triangle = mat.lower_triangle();
        let upper_triangle = mat.upper_triangle();
        Self {
            diag,
            upper_triangle,
            lower_triangle,
        }
    }
}
*/

// TODO probably should do a multilevel gauss seidel and figure out to
// use the same code as L1
//      - test spd of precon again
pub struct Multilevel {
    pub hierarchy: Hierarchy,
    forward_smoothers: Vec<Solver>,
    backward_smoothers: Option<Vec<Solver>>,
}

impl LinearOperator for Multilevel {
    fn apply_mut(&self, r: &mut DVector<f64>) {
        self.init_w_cycle(r);
    }
    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut vec_mut = vec.clone();
        self.apply_mut(&mut vec_mut);
        vec_mut
    }
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        self.apply_mut(out_vec);
    }
}

impl Multilevel {
    // TODO builder pattern for Multilevel
    pub fn new(
        hierarchy: Hierarchy,
        fine_smoother: Option<Rc<dyn LinearOperator>>,
        solve_coarsest_exactly: bool,
        smoother: SmootherType,
        sweeps: usize,
    ) -> Self {
        let fine_mat = hierarchy.get_mat(0);
        // TODO make this an arg?
        let stationary = IterativeMethod::StationaryIteration;

        let build_smoother = |mat: Rc<CsrMatrix<f64>>,
                              smoother: SmootherType,
                              transpose: bool|
         -> Rc<dyn LinearOperator> {
            match smoother {
                SmootherType::L1 => Rc::new(L1::new(&mat)),
                SmootherType::GaussSeidel => {
                    if transpose {
                        Rc::new(BackwardGaussSeidel::new(mat))
                    } else {
                        Rc::new(ForwardGaussSeidel::new(mat))
                    }
                }
            }
        };
        let fine_smoother =
            fine_smoother.unwrap_or(build_smoother(fine_mat.clone(), smoother, false));
        let zeros = DVector::from(vec![0.0; fine_mat.ncols()]);
        let forward_solver = Solver::Iterative(
            Iterative::new(fine_mat.clone(), Some(zeros.clone()))
                .with_solver(stationary)
                .with_max_iter(sweeps)
                .with_preconditioner(fine_smoother)
                .with_tolerance(1e-12),
        );
        let mut forward_smoothers: Vec<Solver> = vec![forward_solver];
        let mut backward_smoothers = match smoother {
            SmootherType::L1 => None,
            SmootherType::GaussSeidel => {
                let backward_smoother = Rc::new(BackwardGaussSeidel::new(fine_mat.clone()));
                let backward_solver = Solver::Iterative(
                    Iterative::new(fine_mat.clone(), Some(zeros))
                        .with_solver(stationary)
                        .with_max_iter(sweeps)
                        .with_preconditioner(backward_smoother)
                        .with_tolerance(1e-12),
                );
                Some(vec![backward_solver])
            }
        };

        let coarse_index = hierarchy.get_matrices().len() - 1;
        forward_smoothers.extend(hierarchy.get_matrices().iter().enumerate().map(|(i, mat)| {
            let smoother = build_smoother(mat.clone(), smoother, false);
            let zeros = DVector::from(vec![0.0; mat.ncols()]);

            let solver: Solver;
            if i == coarse_index && solve_coarsest_exactly {
                solver = Solver::Direct(Direct::new(&mat));
            } else {
                solver = Solver::Iterative(
                    Iterative::new(mat.clone(), Some(zeros))
                        .with_solver(stationary)
                        .with_max_iter(sweeps)
                        .with_preconditioner(smoother)
                        .with_tolerance(0.0),
                );
            }
            solver
        }));

        if let Some(ref mut smoothers) = &mut backward_smoothers {
            smoothers.extend(
                hierarchy
                    .get_matrices()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i < coarse_index)
                    .map(|(_, mat)| {
                        let smoother = build_smoother(mat.clone(), smoother, true);
                        let zeros = DVector::from(vec![0.0; mat.ncols()]);

                        Solver::Iterative(
                            Iterative::new(mat.clone(), Some(zeros))
                                .with_solver(stationary)
                                .with_max_iter(sweeps)
                                .with_preconditioner(smoother)
                                .with_tolerance(0.0),
                        )
                    }),
            );
        }

        Self {
            hierarchy,
            forward_smoothers,
            backward_smoothers,
        }
    }

    pub fn get_hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    fn init_w_cycle(&self, r: &mut DVector<f64>) {
        let mut v = DVector::from(vec![0.0; r.nrows()]);
        if self.hierarchy.levels() > 2 {
            self.w_cycle_recursive(&mut v, &*r, 0, 1);
        } else {
            self.w_cycle_recursive(&mut v, &*r, 0, 1);
        }
        r.copy_from(&v);
    }

    fn w_cycle_recursive(&self, v: &mut DVector<f64>, f: &DVector<f64>, level: usize, mu: usize) {
        let levels = self.hierarchy.levels() - 1;
        if level == levels {
            let solution = self.forward_smoothers[level].solve_with_guess(f, v);
            v.copy_from(&solution);
        } else {
            let p = self.hierarchy.get_partition(level);
            let pt = self.hierarchy.get_interpolation(level);
            let a = &*self.hierarchy.get_mat(level);

            let solution = self.forward_smoothers[level].solve_with_guess(f, v);
            v.copy_from(&solution);

            let f_coarse = spmm(pt, &(f - &spmm(a, v)));
            let mut v_coarse = DVector::from(vec![0.0; f_coarse.nrows()]);
            for _ in 0..mu {
                self.w_cycle_recursive(&mut v_coarse, &f_coarse, level + 1, mu);
                /*
                if level + 1 == levels {
                    break;
                }
                */
            }

            let interpolated = spmm(p, &v_coarse);
            *v += interpolated;
            let solution = if let Some(backward_smoothers) = &self.backward_smoothers {
                backward_smoothers[level].solve_with_guess(f, v)
            } else {
                self.forward_smoothers[level].solve_with_guess(f, v)
            };
            v.copy_from(&solution);
        }
    }
}

// clone should be fine here since everything is Rc
#[derive(Clone)]
pub struct Composite {
    mat: Rc<CsrMatrix<f64>>,
    components: Vec<Rc<Multilevel>>,
}

impl LinearOperator for Composite {
    fn apply_mut(&self, vec: &mut DVector<f64>) {
        self.apply_multiplicative(vec);
    }

    fn apply(&self, vec: &DVector<f64>) -> DVector<f64> {
        let mut vec = vec.clone();
        self.apply_multiplicative(&mut vec);
        vec
    }

    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>) {
        out_vec.copy_from(in_vec);
        self.apply_multiplicative(out_vec);
    }
}

impl Composite {
    pub fn new(mat: Rc<CsrMatrix<f64>>) -> Self {
        Self {
            mat,
            components: Vec::new(),
        }
    }

    pub fn new_with_components(mat: Rc<CsrMatrix<f64>>, components: Vec<Rc<Multilevel>>) -> Self {
        Self { mat, components }
    }

    fn apply_multiplicative(&self, v: &mut DVector<f64>) {
        let mut x = DVector::from(vec![0.0; v.nrows()]);
        let mut r = v.clone();
        let num_steps = 1;
        for component in self.components.iter() {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                r = &*v - spmm(&self.mat, &x);
            }
        }
        for component in self.components.iter().rev().skip(1) {
            for _ in 0..num_steps {
                x = x + component.apply(&r);
                r = &*v - spmm(&self.mat, &x);
            }
        }
        v.copy_from(&x);
    }

    pub fn push(&mut self, component: Rc<Multilevel>) {
        self.components.push(component);
    }

    pub fn rm_oldest(&mut self) {
        self.components.remove(0);
    }

    pub fn components(&self) -> &Vec<Rc<Multilevel>> {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Vec<Rc<Multilevel>> {
        &mut self.components
    }
}
