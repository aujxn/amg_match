use std::{collections::BTreeSet, fs::File, sync::Arc, time::Duration};

use amg_match::{
    adaptive::AdaptiveBuilder,
    hierarchy::Hierarchy,
    interpolation::{classical, classical_step, InterpolationType},
    preconditioner::{BlockSmootherType, LinearOperator, SmootherType, L1},
    solver::{Iterative, IterativeMethod, LogInterval},
    utils::{load_system, normalize},
    Vector,
};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

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

    let epsilon = 1e-12;
    //let mut results: Vec<(usize, usize)> = Vec::new();
    let method = IterativeMethod::StationaryIteration;
    //let method = IterativeMethod::ConjugateGradient;
    let i = 2;

    let prefix = "data/laplace";
    let name = format!("{}", i);
    let (mat, b, _coords, _rbms, projector) = load_system(&prefix, &name, false);
    let bdy_map = projector.transpose_view();
    let dim = mat.rows();
    //let cf = dim as f64 / 100.0;
    let coarsening_factor = 8.0;
    //let rand: Vector = Vector::from_element(b.len(), 1.0).normalize();

    let fine_l1 = Arc::new(L1::new(&mat));
    let guess: Vector = Vector::from_elem(dim, 1.0);
    let stationary = Iterative::new(mat.clone(), Some(guess))
        .with_solver(IterativeMethod::StationaryIteration)
        .with_max_iter(10)
        .with_preconditioner(fine_l1.clone());
    let zeros = Vector::from(vec![0.0; dim]);
    let mut near_null: Vector = stationary.apply(&zeros);
    normalize(&mut near_null, &mat);

    let mut buffer = File::create("viz/smooth_near_null.npz").unwrap();
    let p_near_null = &bdy_map * &near_null;
    write_raw_gf(p_near_null.as_slice().unwrap(), &mut buffer).unwrap();

    let mut hierarchy = Hierarchy::new(mat.clone());
    let interpolation_type = InterpolationType::Classical;
    let coarse_near_null = hierarchy.add_level(&near_null, coarsening_factor, interpolation_type);

    let target_size = (dim as f64 / coarsening_factor).ceil() as usize;
    let (_coarse_near_null, _r, p, _ac, all_c_dofs, fine_dofs) =
        classical_step(&mat, &near_null, &None, target_size);

    let mut indices_vector = Vector::from_elem(dim, -1.0);
    for i in fine_dofs {
        indices_vector[i] = 0.0;
    }
    for (stage, group) in all_c_dofs.iter().rev().enumerate() {
        for i in group.iter().copied() {
            indices_vector[i] = stage as f64 + 1.0;
        }
    }
    let indices_bdy = &bdy_map * &indices_vector;
    let mut buffer = File::create("viz/c_points.npz").unwrap();
    write_raw_gf(indices_bdy.as_slice().unwrap(), &mut buffer).unwrap();

    let pt = p.transpose_view().to_csr();
    for (i, row) in pt.outer_iterator().enumerate() {
        let mut basis = Vector::from_elem(dim, 0.0);
        for group in all_c_dofs.iter() {
            for i in group.iter().copied() {
                basis[i] = 2.0;
            }
        }
        for (j, v) in row.iter() {
            basis[j] = *v;
        }
        let basis = &bdy_map * &basis;
        let mut buffer = File::create(format!("viz/basis_coarse_{}.npz", i)).unwrap();
        write_raw_gf(basis.as_slice().unwrap(), &mut buffer).unwrap();
    }
    //let (coarse_near_null, r, p, mut mat_coarse) = classical(&mat, &near_null);
    //

    let all_coarse: BTreeSet<usize> =
        all_c_dofs
            .iter()
            .fold(BTreeSet::new(), |mut acc, coarse_set| {
                acc.extend(coarse_set);
                acc
            });
    let fine_indices: Vec<usize> = all_coarse.into_iter().collect();

    for (i, row) in p.outer_iterator().enumerate() {
        let mut basis = Vector::from_elem(dim, 0.0);
        basis[i] = 2.0;
        /*
        for group in all_c_dofs.iter() {
            for i in group.iter().copied() {
                basis[i] = 2.0;
            }
        }
        */
        for (j, v) in row.iter() {
            basis[fine_indices[j]] = *v;
        }
        let basis = &bdy_map * &basis;
        let mut buffer = File::create(format!("viz/basis_fine_{}.npz", i)).unwrap();
        write_raw_gf(basis.as_slice().unwrap(), &mut buffer).unwrap();
    }
}
