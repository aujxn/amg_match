//! Definition of the `LinearOperator` trait as well as implementors
//! of said trait.

use crate::parallel_ops::spmm;
use crate::partitioner::Hierarchy;
use crate::solver::{IterativeMethod, IterativeSolver};
use crate::utils::{inner_product, norm};
use nalgebra::base::DVector;
use nalgebra_sparse::CsrMatrix;
use std::{cell::RefCell, rc::Rc};

pub trait LinearOperator {
    fn apply_mut(&self, vec: &mut DVector<f64>);
    fn apply(&self, vec: &DVector<f64>) -> DVector<f64>;
    fn apply_input(&self, in_vec: &DVector<f64>, out_vec: &mut DVector<f64>);
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

/*
pub struct ForwardGaussSeidel {
    lower_triangle: CsrMatrix<f64>,
}

impl LinearOperator for ForwardGaussSeidel {
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

impl LinearOperator for BackwardGaussSeidel {
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
    forward_smoothers: Vec<IterativeSolver>,
    //coarse_solver: Rc<dyn LinearOperator>,
    //_backward_smoothers: Vec<Rc<dyn LinearOperator>>,  // Eventually should probably support non-symmetric smoothers
}

/*
struct MultilevelWorkspace {
    x_ks: Vec<DVector<f64>>,
    b_ks: Vec<DVector<f64>>,
}

impl MultilevelWorkspace {
    fn new() -> Self {
        Self {
            x_ks: Vec::new(),
            b_ks: Vec::new(),
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
            }
        }

        let mut sizes: Vec<usize> = vec![hierarchy.get_mat(0).nrows()];
        sizes.extend(hierarchy.get_matrices().iter().map(|p| p.nrows()));

        for (i, size) in sizes.into_iter().enumerate() {
            ws.x_ks[i].resize_vertically_mut(size, 0.0);
            ws.b_ks[i].resize_vertically_mut(size, 0.0);
        }
    });
}
*/

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
    // TODO different cycle types
    pub fn new(hierarchy: Hierarchy, fine_smoother: Option<Rc<dyn LinearOperator>>) -> Self {
        let fine_mat = hierarchy.get_mat(0);
        let base_steps = 3;
        let stationary = IterativeMethod::StationaryIteration;

        let smoother = fine_smoother.unwrap_or(Rc::new(L1::new(&fine_mat)));
        let zeros = DVector::from(vec![0.0; fine_mat.ncols()]);
        let solver = IterativeSolver::new(fine_mat.clone(), Some(zeros))
            .with_solver(stationary)
            .with_max_iter(base_steps)
            //.with_max_iter(5)
            .with_preconditioner(smoother)
            .with_tolerance(1e-12);
        let mut forward_smoothers: Vec<IterativeSolver> = vec![solver];

        //let coarse_index = hierarchy.get_matrices().len() - 1;
        forward_smoothers.extend(hierarchy.get_matrices().iter().enumerate().map(|(i, mat)| {
            // Geometric increase in smoother step count down hierarchy
            //let mut steps: usize = (base_steps as u32 * 2u32.pow(i as u32 + 1)) as usize;
            let steps = base_steps;
            let smoother = Rc::new(L1::new(&mat));
            let zeros = DVector::from(vec![0.0; mat.ncols()]);
            let method = stationary;
            let tolerance = 1e-12;
            /*
            if i == coarse_index {
                steps = 5000;
                method = IterativeMethod::ConjugateGradient;
                tolerance = 1e-6;
            }
            */
            let solver: IterativeSolver = IterativeSolver::new(mat.clone(), Some(zeros))
                .with_solver(method)
                .with_max_iter(steps)
                .with_preconditioner(smoother)
                .with_tolerance(tolerance);
            solver
        }));

        //let coarse_solver = Rc::new(forward_smoothers.pop().unwrap());

        Self {
            hierarchy,
            forward_smoothers,
            //coarse_solver,
        }
    }

    pub fn get_hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    fn init_w_cycle(&self, r: &mut DVector<f64>) {
        let mut v = DVector::from(vec![0.0; r.nrows()]);
        self.w_cycle_recursive(&mut v, &*r, 0, 1);
        r.copy_from(&v);
    }

    fn w_cycle_recursive(&self, v: &mut DVector<f64>, f: &DVector<f64>, level: usize, mu: usize) {
        let levels = self.hierarchy.levels() - 1;
        if level == levels {
            //let solution = self.coarse_solver.apply(f);
            let (solution, _) = self.forward_smoothers[level].solve_with_guess(f, v);

            /*
            // check if converged without using SolveInfo / LogInterval
            let coarse_mat = self.hierarchy.get_matrices().last().unwrap();
            let coarse_residual = f - &**coarse_mat * &solution;
            //let coarse_residual_norm = norm(&coarse_residual, coarse_mat);
            //let initial_residual_norm = norm(f, coarse_mat);
            let coarse_residual_norm = coarse_residual.norm();
            let initial_residual_norm = f.norm();
            let relative_norm = coarse_residual_norm / initial_residual_norm;
            if relative_norm > 1e-4 {
                warn!(
                    "Coarse solver didn't converge with relative norm: {:.2e}",
                    relative_norm
                );
            }
            */

            v.copy_from(&solution);
        } else {
            let p = self.hierarchy.get_partition(level);
            let pt = self.hierarchy.get_interpolation(level);
            let a = &*self.hierarchy.get_mat(level);

            let (solution, _) = self.forward_smoothers[level].solve_with_guess(f, v);
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
            let (solution, _) = self.forward_smoothers[level].solve_with_guess(f, v);
            v.copy_from(&solution);
        }
    }

    /*
    fn v_cycle(&self, r: &mut DVector<f64>) {
        resize_ml_workspace(&self.hierarchy);
        ML_WORKSPACE.with(|ws| {
            let ws = &mut *ws.borrow_mut();

            let levels = self.hierarchy.levels() - 1;
            let p_ks = self.hierarchy.get_partitions();
            let pt_ks = self.hierarchy.get_interpolations();

            // reset the workspaces
            for (xk, bk) in ws.x_ks.iter_mut().zip(ws.b_ks.iter_mut()) {
                xk.fill(0.0);
                bk.fill(0.0);
            }

            ws.b_ks[0].copy_from(&*r);

            for level in 0..levels {
                ws.x_ks[level].copy_from(&ws.b_ks[level]);
                self.forward_smoothers[level].apply_mut(&mut ws.x_ks[level]);
                ws.b_ks[level + 1] = spmm(
                    &pt_ks[level],
                    &(&ws.b_ks[level] - spmm(&self.hierarchy.get_mat(level), &ws.x_ks[level])),
                );
            }

            let solution = self.coarse_solver.apply(&ws.b_ks[levels]);

            // check if converged without using SolveInfo / LogInterval
            let coarse_mat = self.hierarchy.get_matrices().last().unwrap();
            let coarse_residual = &ws.b_ks[levels] - &**coarse_mat * &solution;
            let coarse_residual_norm = norm(&coarse_residual, coarse_mat);
            let initial_residual_norm = norm(&ws.b_ks[levels], coarse_mat);
            let relative_norm = coarse_residual_norm / initial_residual_norm;
            if relative_norm > 1e-4 {
                warn!(
                    "Coarse solver didn't converge with relative norm: {:.2e}",
                    relative_norm
                );
            }

            ws.x_ks[levels].copy_from(&solution);

            for level in (0..levels).rev() {
                let interpolated = spmm(&p_ks[level], &ws.x_ks[level + 1]);
                ws.x_ks[level] += interpolated;

                let mut r =
                    &ws.b_ks[level] - spmm(&*self.hierarchy.get_mat(level), &ws.x_ks[level]);
                self.forward_smoothers[level].apply_mut(&mut r);
                ws.x_ks[level] += &r
            }
            r.copy_from(&ws.x_ks[0]);
        });
    }
    */
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

// clone should be fine here since everything is Rc
#[derive(Clone)]
pub struct Composite {
    mat: Rc<CsrMatrix<f64>>,
    components: Vec<Rc<dyn LinearOperator>>,
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

    pub fn new_with_components(
        mat: Rc<CsrMatrix<f64>>,
        components: Vec<Rc<dyn LinearOperator>>,
    ) -> Self {
        Self { mat, components }
    }

    fn apply_multiplicative(&self, v: &mut DVector<f64>) {
        let mut x = DVector::from(vec![0.0; v.nrows()]);
        let mut r = v.clone();
        for component in self.components.iter() {
            x = x + component.apply(&r);
            r = &*v - spmm(&self.mat, &x);
        }
        for component in self.components.iter().rev().skip(1) {
            x = x + component.apply(&r);
            r = &*v - spmm(&self.mat, &x);
        }
        v.copy_from(&x);
    }
    /*
    fn apply_multiplicative(&self, r: &mut DVector<f64>) {
        COMPOSITE_WORKSPACE.with(|ws| {
            let ws = &mut *ws.borrow_mut();
            let dim = r.nrows();
            ws.x.resize_vertically_mut(dim, 0.0);
            ws.y.resize_vertically_mut(dim, 0.0);
            ws.r_work.resize_vertically_mut(dim, 0.0);

            ws.x.fill(0.0);
            ws.y.fill(0.0);
            ws.r_work.copy_from(r);

            /*
            if self.components().len() == 1 {
                self.components()[0].apply_mut(r);
                return;
            }
            */

            for component in self.components.iter() {
                ws.y.copy_from(&ws.r_work);
                component.apply_mut(&mut ws.y);
                ws.x += &ws.y;
                // *r -= self.mat * &y;
                ws.r_work = &*r - &*self.mat * &ws.x;
                //ws.r_work.copy_from(r);
                //spmm_csr_dense(1.0, &mut ws.r_work, -1.0, &self.mat, &ws.x);
            }
            for component in self.components.iter().rev() {
                ws.y.copy_from(&ws.r_work);
                component.apply_mut(&mut ws.y);
                ws.x += &ws.y;
                // *r -= self.mat * &y;
                ws.r_work = &*r - &*self.mat * &ws.x;
                //ws.r_work.copy_from(r);
                //spmm_csr_dense(1.0, &mut ws.r_work, -1.0, &self.mat, &ws.x);
            }
            r.copy_from(&ws.x);
        });
    }
    */

    pub fn push(&mut self, component: Rc<dyn LinearOperator>) {
        self.components.push(component);
    }

    pub fn rm_oldest(&mut self) {
        self.components.remove(0);
    }

    pub fn components(&self) -> &Vec<Rc<dyn LinearOperator>> {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Vec<Rc<dyn LinearOperator>> {
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
        let application = CompositeType::Multiplicative;
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
