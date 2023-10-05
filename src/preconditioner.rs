//! Definition of the `Preconditioner` trait as well as implementors
//! of said trait.

use crate::io::CompositeData;
use crate::parallel_ops::{interpolate, spmm_csr_dense};
use crate::partitioner::Hierarchy;
use crate::solver::{lsolve, pcg, stationary, usolve};
use nalgebra::base::DVector;
use nalgebra_sparse::CsrMatrix;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::{cell::RefCell, path::Path, rc::Rc};

// Maybe PC trait should be generalized to LinearOperator...
// Then could have 3 options for apply:
// - one in place
// - one returning vec
// - one with input and output vec
//
// Really a PC is just a LO. Also solvers are LO. So having
// a trait called PC is redundant.
pub trait Preconditioner {
    fn apply(&self, r: &mut DVector<f64>);
}

#[derive(Serialize, Deserialize)]
pub struct Identity;
impl Identity {
    fn new() -> Self {
        Self
    }
}

impl Preconditioner for Identity {
    fn apply(&self, r: &mut DVector<f64>) {}
}

//#[derive(Serialize, Deserialize)]
pub struct L1 {
    l1_inverse: DVector<f64>,
}

impl Preconditioner for L1 {
    fn apply(&self, r: &mut DVector<f64>) {
        r.component_mul_assign(&self.l1_inverse);
    }
}

impl L1 {
    pub fn new(mat: &CsrMatrix<f64>) -> Self {
        let l1_inverse: Vec<f64> = mat
            .row_iter()
            .map(|row_vec| {
                let row_sum_abs: f64 = row_vec.values().iter().map(|val| val.abs()).sum();
                1.0 / row_sum_abs
            })
            .collect();
        let l1_inverse: DVector<f64> = DVector::from(l1_inverse);
        Self { l1_inverse }
    }
}

pub struct PcgL1 {
    mat: Rc<CsrMatrix<f64>>,
    steps: usize,
    smoother: L1,
}

impl PcgL1 {
    pub fn new(mat: Rc<CsrMatrix<f64>>, steps: usize) -> Self {
        let smoother = L1::new(&mat);
        Self {
            mat,
            steps,
            smoother,
        }
    }
}

type Solver = fn(
    &CsrMatrix<f64>,
    &DVector<f64>,
    &mut DVector<f64>,
    usize,
    f64,
    &dyn Preconditioner,
    Option<usize>,
) -> (bool, f64);

pub struct Smoother<P> {
    solver: Solver,
    matrix: Rc<CsrMatrix<f64>>,
    preconditioner: P,
    steps: usize,
}

impl<P: Preconditioner> Smoother<P> {
    pub fn new(
        solver: Solver,
        matrix: Rc<CsrMatrix<f64>>,
        preconditioner: P,
        steps: usize,
    ) -> Self {
        Self {
            solver,
            matrix,
            preconditioner,
            steps,
        }
    }
}

impl<P: Preconditioner> Preconditioner for Smoother<P> {
    fn apply(&self, r: &mut DVector<f64>) {
        let mut x = DVector::from(vec![0.0; r.len()]);
        (self.solver)(
            &self.matrix,
            r,
            &mut x,
            self.steps,
            1e-12,
            &self.preconditioner,
            None,
        );
        *r = x;
    }
}

impl Preconditioner for PcgL1 {
    fn apply(&self, r: &mut DVector<f64>) {
        let mut x = DVector::from(vec![0.0; r.nrows()]);
        let _ = pcg(
            &self.mat,
            r,
            &mut x,
            self.steps,
            1e-12,
            &self.smoother,
            None,
        );
        r.copy_from(&x);
    }
}

pub struct ForwardGaussSeidel {
    lower_triangle: CsrMatrix<f64>,
}

impl Preconditioner for ForwardGaussSeidel {
    fn apply(&self, r: &mut DVector<f64>) {
        lsolve(&self.lower_triangle, r);
    }
}

impl ForwardGaussSeidel {
    pub fn new(mat: &CsrMatrix<f64>) -> ForwardGaussSeidel {
        let lower_triangle = mat.lower_triangle();
        Self { lower_triangle }
    }
}

pub struct BackwardGaussSeidel {
    upper_triangle: CsrMatrix<f64>,
}

impl Preconditioner for BackwardGaussSeidel {
    fn apply(&self, r: &mut DVector<f64>) {
        usolve(&self.upper_triangle, r);
    }
}

impl BackwardGaussSeidel {
    pub fn new(mat: &CsrMatrix<f64>) -> BackwardGaussSeidel {
        let upper_triangle = mat.upper_triangle();
        Self { upper_triangle }
    }
}

pub struct SymmetricGaussSeidel {
    diag: DVector<f64>,
    upper_triangle: CsrMatrix<f64>,
    lower_triangle: CsrMatrix<f64>,
}

impl Preconditioner for SymmetricGaussSeidel {
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

// TODO probably should do a multilevel gauss seidel and figure out to
// use the same code as L1
//      - test spd of precon again
pub struct Multilevel<T> {
    hierarchy: Hierarchy,
    forward_smoothers: Vec<T>,
    //_backward_smoothers: Vec<T>,  // Eventually should probably support non-symmetric smoothers
}

struct MultilevelWorkspace {
    x_ks: Vec<DVector<f64>>,
    b_ks: Vec<DVector<f64>>,
    r_ks: Vec<DVector<f64>>,
}

impl MultilevelWorkspace {
    fn new() -> Self {
        Self {
            x_ks: Vec::new(),
            b_ks: Vec::new(),
            r_ks: Vec::new(),
        }
    }
}

// This is bad but since we are singlethreaded from view of the V-Cycle it's fine for now
thread_local!(static ML_WORKSPACE: RefCell<MultilevelWorkspace> = RefCell::new(MultilevelWorkspace::new()));

fn resize_ml_workspace(hierarchy: &Hierarchy) {
    ML_WORKSPACE.with(|ws| {
        let ws = &mut *ws.borrow_mut();
        if ws.x_ks.len() < hierarchy.levels() {
            let new_levels = hierarchy.levels() - ws.x_ks.len();
            for _ in 0..new_levels {
                ws.x_ks.push(DVector::from(Vec::new()));
                ws.b_ks.push(DVector::from(Vec::new()));
                ws.r_ks.push(DVector::from(Vec::new()));
            }
        }

        let mut sizes: Vec<usize> = vec![hierarchy.get_mat(0).nrows()];
        sizes.extend(hierarchy.get_matrices().iter().map(|p| p.nrows()));

        for (i, size) in sizes.into_iter().enumerate() {
            ws.x_ks[i].resize_vertically_mut(size, 0.0);
            ws.r_ks[i].resize_vertically_mut(size, 0.0);
            ws.b_ks[i].resize_vertically_mut(size, 0.0);
        }
    });
}

impl<T: Preconditioner> Preconditioner for Multilevel<T> {
    fn apply(&self, r: &mut DVector<f64>) {
        resize_ml_workspace(&self.hierarchy);
        ML_WORKSPACE.with(|ws| {
            let ws = &mut *ws.borrow_mut();

            let levels = self.hierarchy.levels() - 1;
            let p_ks = self.hierarchy.get_partitions();
            let pt_ks = self.hierarchy.get_interpolations();

            // reset the workspaces
            for ((xk, bk), rk) in ws
                .x_ks
                .iter_mut()
                .zip(ws.b_ks.iter_mut())
                .zip(ws.r_ks.iter_mut())
            {
                xk.fill(0.0);
                bk.fill(0.0);
                rk.fill(0.0);
            }

            ws.b_ks[0].copy_from(&*r);

            for level in 0..levels {
                //for _ in 0..self.smoothing_steps {
                ws.r_ks[level].copy_from(&ws.b_ks[level]);
                crate::parallel_ops::spmm_csr_dense(
                    1.0,
                    &mut ws.r_ks[level],
                    -1.0,
                    &self.hierarchy.get_mat(level),
                    &ws.x_ks[level],
                );

                self.forward_smoothers[level].apply(&mut ws.r_ks[level]);
                ws.x_ks[level] += &ws.r_ks[level];
                //}

                //let r_k = &self.b_ks[level] - &(&mat_ks[level] * &self.x_ks[level]);
                ws.r_ks[level].copy_from(&ws.b_ks[level]);
                spmm_csr_dense(
                    1.0,
                    &mut ws.r_ks[level],
                    -1.0,
                    &self.hierarchy.get_mat(level),
                    &ws.x_ks[level],
                );
                //let p_t = p_ks[level + 1].transpose();
                //self.b_ks[level + 1] = &p_t * &r_k;
                spmm_csr_dense(
                    0.0,
                    &mut ws.b_ks[level + 1],
                    1.0,
                    &pt_ks[level],
                    &ws.r_ks[level],
                );
            }

            // Solve the coarsest problem almost exactly
            let (converged, ratio) = pcg(
                //let (converged, ratio) = stationary(
                &self.hierarchy.get_mat(levels),
                &ws.b_ks[levels],
                &mut ws.x_ks[levels],
                300,
                1.0e-6,
                &self.forward_smoothers[levels],
                //&Identity::new(),
                //&L1::new(&self.hierarchy.get_mat(levels)),
                None, //Some(5),
            );

            if !converged {
                warn!(
                    "solver didn't converge on coarsest level with final ratio: {:3e}",
                    ratio
                );
            }

            for level in (0..levels).rev() {
                //let interpolated_x = self.hierarchy.get_partition(level + 1) * &self.x_ks[level + 1];
                //self.x_ks[level] += &interpolated_x;
                interpolate(&mut ws.r_ks[level], &ws.x_ks[level + 1], &p_ks[level]);
                ws.x_ks[level] += &ws.r_ks[level];

                //for _ in 0..self.smoothing_steps {
                //let mut r_k = &self.b_ks[level] - &(self.hierarchy.get_matrix(level) * &self.x_ks[level]);
                ws.r_ks[level].copy_from(&ws.b_ks[level]);
                spmm_csr_dense(
                    1.0,
                    &mut ws.r_ks[level],
                    -1.0,
                    &self.hierarchy.get_mat(level),
                    &ws.x_ks[level],
                );

                self.forward_smoothers[level].apply(&mut ws.r_ks[level]);
                ws.x_ks[level] += &ws.r_ks[level];
                //}
            }
            r.copy_from(&ws.x_ks[0]);
        });
    }
}

/*
impl TwoLevel {
    pub fn new(hierarchy: Hierarchy) -> Self {

    }
}
*/

impl Multilevel<Smoother<L1>> {
    // TODO this constructor should borrow mat when partitioner/hierarchy changes happen
    pub fn new(hierarchy: Hierarchy) -> Self {
        let fine_mat = hierarchy.get_mat(0);
        let base_steps = 3;
        let mut forward_smoothers: Vec<Smoother<L1>> = vec![Smoother::new(
            stationary,
            fine_mat.clone(),
            L1::new(&fine_mat),
            base_steps as usize,
        )];
        forward_smoothers.extend(
            hierarchy
                .get_matrices()
                .iter()
                .enumerate()
                .map(|(i, mat)| {
                    let mut steps: usize = (base_steps * 2u32.pow(i as u32 + 1)) as usize;
                    if i == hierarchy.get_matrices().len() - 1 {
                        steps = 1;
                    }
                    Smoother::new(stationary, mat.clone(), L1::new(&mat), steps)
                })
                .collect::<Vec<_>>(),
        );

        Self {
            hierarchy,
            forward_smoothers,
        }
    }

    pub fn get_hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }
}

struct CompositeWorkspace {
    x: DVector<f64>,
    y: DVector<f64>,
    r_work: DVector<f64>,
}

impl CompositeWorkspace {
    fn new() -> Self {
        Self {
            x: DVector::from(Vec::new()),
            y: DVector::from(Vec::new()),
            r_work: DVector::from(Vec::new()),
        }
    }
}

// This is bad but since we are singlethreaded from view of the V-Cycle it's fine for now
thread_local!(static COMPOSITE_WORKSPACE: RefCell<CompositeWorkspace> = RefCell::new(CompositeWorkspace::new()));

pub struct Composite<T> {
    mat: Rc<CsrMatrix<f64>>,
    components: Vec<T>,
    application: CompositeType,
}

pub enum CompositeType {
    Additive, //TODO
    Sequential,
}

impl<T: Preconditioner> Preconditioner for Composite<T> {
    fn apply(&self, r: &mut DVector<f64>) {
        COMPOSITE_WORKSPACE.with(|ws| {
            let ws = &mut *ws.borrow_mut();
            let dim = r.nrows();
            ws.x.resize_vertically_mut(dim, 0.0);
            ws.y.resize_vertically_mut(dim, 0.0);
            ws.r_work.resize_vertically_mut(dim, 0.0);

            ws.x.fill(0.0);
            ws.r_work.copy_from(r);
        });

        match self.application {
            CompositeType::Sequential => self.apply_sequential(r),
            CompositeType::Additive => unimplemented!(), //self.apply_additive(r),
        }
    }
}

impl<T: Preconditioner> Composite<T> {
    pub fn new(mat: Rc<CsrMatrix<f64>>, components: Vec<T>, application: CompositeType) -> Self {
        Self {
            mat,
            components,
            application,
        }
    }

    fn apply_sequential(&self, r: &mut DVector<f64>) {
        COMPOSITE_WORKSPACE.with(|ws| {
            let ws = &mut *ws.borrow_mut();

            for component in self.components.iter() {
                ws.y.copy_from(&ws.r_work);
                component.apply(&mut ws.y);
                ws.x += &ws.y;
                //*r -= self.mat * &y;
                ws.r_work.copy_from(r);
                spmm_csr_dense(1.0, &mut ws.r_work, -1.0, &self.mat, &ws.x);
            }
            for component in self.components.iter().rev() {
                ws.y.copy_from(&ws.r_work);
                component.apply(&mut ws.y);
                ws.x += &ws.y;
                //*r -= self.mat * &y;
                ws.r_work.copy_from(r);
                spmm_csr_dense(1.0, &mut ws.r_work, -1.0, &self.mat, &ws.x);
            }
            r.copy_from(&ws.x);
        });
    }

    /*
    fn apply_additive(&self, r: &mut DVector<f64>) {
        let num_components = self.components.len() as f64;
        for component in self.components.iter_mut() {
            self.y.copy_from(&self.r_work);
            component.apply(&self.y);
            self.x += &self.y;
        }
        self.x /= num_components;
        r.copy_from(&self.x);
    }
    */

    pub fn push(&mut self, component: T) {
        self.components.push(component);
    }

    pub fn rm_oldest(&mut self) {
        self.components.remove(0);
    }

    pub fn components(&self) -> &Vec<T> {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Vec<T> {
        &mut self.components
    }
}

/*
impl Composite<Multilevel<Smoother<L1>>> {
    pub fn save<P: AsRef<Path>>(&self, p: P, notes: String) {
        let intermediate = CompositeData::new(&self, notes);
        let serialized = serde_json::to_string(&intermediate).unwrap();
        let mut file = File::create(p).unwrap();
        file.write_all(&serialized.as_bytes()).unwrap();
    }

    pub fn load<P: AsRef<Path>>(mat: Rc<CsrMatrix<f64>>, p: P) -> (Self, String) {
        let application = CompositeType::Sequential;
        let file = File::open(p).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut serialized = String::new();
        buf_reader.read_to_string(&mut serialized).unwrap();
        let deserialized: CompositeData = serde_json::from_str(&serialized).unwrap();
        let components = Vec::new();
        let mut pc = Composite {
            mat: mat.clone(),
            components,
            application,
        };

        for hierarchy in deserialized.hierarchies {
            let partition_matrices: Vec<CsrMatrix<f64>> = hierarchy
                .partition_matrices
                .into_iter()
                .map(|mat| mat.into())
                .collect();
            let hierarchy = Hierarchy::from_hierarchy(mat.clone(), partition_matrices);
            let comp: Multilevel<L1> = Multilevel::new(hierarchy);
            pc.push(comp);
        }
        (pc, deserialized.notes)
    }
}
*/
