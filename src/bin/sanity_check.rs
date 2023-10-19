use amg_match::{
    adaptive::build_adaptive,
    solver::pcg,
    utils::{load_system, random_vec},
};
use nalgebra::DVector;

#[macro_use]
extern crate log;

fn main() {
    pretty_env_logger::init();

    let cf = 2.0;
    let mut results: Vec<(usize, usize, usize, usize)> = Vec::new();

    for i in 0..8 {
        let prefix = format!("data/laplace/{}", i);
        let (mat, b) = load_system(&prefix);
        let dim = mat.nrows();
        info!("Starting: {}", i);

        let (two_level, _) = build_adaptive(mat.clone(), cf, 2, 0.01, 1, &prefix);
        let mut x: DVector<f64> = random_vec(dim);
        let (converged_2, ratio_2, iters_2) =
            pcg(&mat, &b, &mut x, 10000, 1e-6, &two_level, Some(3));
        if !converged_2 {
            warn!("2 level didn't converge with ratio: {}", ratio_2);
        }

        let (multi_level, _) = build_adaptive(mat.clone(), cf, 10, 0.01, 1, &prefix);
        let mut x: DVector<f64> = random_vec(dim);
        let (converged_multi, ratio_multi, iters_multi) =
            pcg(&mat, &b, &mut x, 10000, 1e-6, &multi_level, Some(3));
        if !converged_multi {
            warn!("multi level didn't converge with ratio: {}", ratio_multi);
        }

        results.push((
            dim,
            iters_2,
            iters_multi,
            multi_level.components()[0].get_hierarchy().levels(),
        ));
    }

    println!(
        "{:15} {:15} {:15} {:15}",
        "Dimension", "2-level iters", "multi-level iters", "levels"
    );
    for (dim, two, multi, levels) in results {
        println!("{:15} {:15} {:15} {:15}", dim, two, multi, levels);
    }
}
