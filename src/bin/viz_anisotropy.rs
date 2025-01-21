use amg_match::{
    adaptive::AdaptiveBuilder,
    preconditioner::LinearOperator,
    solver::{Iterative, IterativeMethod, LogInterval},
    utils::load_system,
};
use nalgebra::DVector;
use npyz;
use std::{fs::File, sync::Arc};

#[macro_use]
extern crate log;

fn write_raw_gf(grid_func: &[f64], out_buf: impl std::io::Write) -> std::io::Result<()> {
    use npyz::WriterBuilder;
    let mut writer = {
        npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[grid_func.len() as u64])
            .writer(out_buf)
            .begin_nd()?
    };

    writer.extend(grid_func)?;
    writer.finish()?;
    Ok(())
}

fn main() {
    pretty_env_logger::init();

    let prefix = "data/anisotropy/anisotropy_2d";
    //let prefix = "data/laplace/4";
    let (mat, b, coords, projector) = load_system(prefix);
    let mat_ref = &*mat;
    let dim = mat.nrows();
    //let b: DVector<f64> = DVector::zeros(dim);
    let guess: DVector<f64> = DVector::zeros(dim);
    //let guess: DVector<f64> = random_vec(dim);
    //let guess: DVector<f64> = DVector::from_element(dim, 1.0);

    let max_components = 10;
    let coarsening_factor = 4.0;
    let test_iters = 30;

    let adaptive_builder = AdaptiveBuilder::new(mat.clone())
        .with_max_components(max_components)
        .with_coarsening_factor(coarsening_factor)
        //.with_project_first_only()
        //.with_max_level(2)
        .with_max_test_iters(test_iters);

    let (pc, _convergence_hist, near_nulls) = adaptive_builder.build();
    let pc = Arc::new(pc);

    for (i, near_null) in near_nulls.iter().enumerate() {
        let mut buffer = File::create(format!("near_null-{}.npz", i)).unwrap();
        let near_null = &projector * near_null;
        write_raw_gf(near_null.as_slice(), &mut buffer).unwrap();
    }

    //let fine_l1 = Arc::new(L1::new(&mat));
    //let solver = Iterative::new(mat.clone(), Some(guess))
    let solver = Iterative::new(mat.clone(), Some(guess.clone()))
        .with_solver(IterativeMethod::ConjugateGradient)
        .with_tolerance(1e-12)
        .with_max_iter(300)
        .with_log_interval(LogInterval::Iterations(1))
        //.with_preconditioner(fine_l1.clone());
        .with_preconditioner(pc.clone());
    //.with_preconditioner(pc.components()[2].clone());

    let solution = solver.apply(&b);
    let mut buffer = File::create("solution.npz").unwrap();
    let solution_plot = &projector * &solution;
    write_raw_gf(solution_plot.as_slice(), &mut buffer).unwrap();

    let solver = Iterative::new(mat.clone(), Some(guess.clone()))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_tolerance(1e-12)
        .with_max_iter(8)
        .with_log_interval(LogInterval::Iterations(1))
        //.with_preconditioner(fine_l1.clone());
        .with_preconditioner(pc.clone());

    let approx = solver.apply(&b);
    let error = solution - &approx;
    let error_plot = &projector * &error;
    let mut buffer = File::create("error.npz").unwrap();
    write_raw_gf(error_plot.as_slice(), &mut buffer).unwrap();

    let residual = &b - mat_ref * &approx;
    let residual_plot = &projector * &residual;
    let mut buffer = File::create("residual.npz").unwrap();
    write_raw_gf(residual_plot.as_slice(), &mut buffer).unwrap();

    // restrict linear (or smooth) and interpolate
    // plot columns of p on fine (coarse basis function related to this column)
    // fine function, smooth, restrict pointwise, interpolate (should be similar)
    let h = pc.components()[max_components - 1].get_hierarchy();
    let p = h.get_partition(0);
    let pt = &h.restriction_matrices[0];
    let p_pt_residual = p * (pt * error);
    let p_pt_residual_plot = &projector * p_pt_residual;
    let mut buffer = File::create("p_pt_residual.npz").unwrap();
    write_raw_gf(p_pt_residual_plot.as_slice(), &mut buffer).unwrap();

    let mut x_vals = DVector::zeros(dim);
    for (i, coord) in coords.iter().enumerate() {
        x_vals[i] = coord[0];
    }
    //let mut buffer = File::create("x_vals.npz").unwrap();
    //write_raw_gf(x_vals.as_slice(), &mut buffer).unwrap();
    let coarse_xvals = pt * x_vals;
    let interp_xvals_plot = &projector * (p * coarse_xvals);
    let mut buffer = File::create("interp_x_vals.npz").unwrap();
    write_raw_gf(interp_xvals_plot.as_slice(), &mut buffer).unwrap();

    let p = h.get_partition(0);
    let pt = p.transpose();

    for (i, basis_func) in pt.row_iter().enumerate().take(50) {
        let mut basis = DVector::zeros(dim);
        for (j, val) in basis_func
            .col_indices()
            .iter()
            .zip(basis_func.values().iter())
        {
            basis[*j] = *val;
        }
        let basis_plot = &projector * &basis;
        let mut buffer = File::create(format!("basis-{}.npz", i)).unwrap();
        write_raw_gf(basis_plot.as_slice(), &mut buffer).unwrap();
    }

    /*
    let near_null_space = DMatrix::zeros(n, max_components);
    for (j, vec) in near_nulls.iter().enumerate() {
        near_null_space.set_column(j, data);
    }
    */
}
