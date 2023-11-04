use amg_match::{
    adaptive::build_adaptive,
    solver::{pcg, stationary},
    utils::{load_system, random_vec},
};
use nalgebra::DVector;

#[macro_use]
extern crate log;

fn main() {
    pretty_env_logger::init();

    let cf = 2.0;
    let epsilon = 1e-6;
    let mut results: Vec<(usize, usize, usize, usize)> = Vec::new();

    for i in 0..9 {
        let prefix = format!("data/laplace/{}", i);
        let (mat, _b) = load_system(&prefix);
        let dim = mat.nrows();
        info!("Starting: {}", i);

        let rand: DVector<f64> = random_vec(dim);
        let b: DVector<f64> = random_vec(dim);
        //let rand: DVector<f64> = DVector::from_element(b.len(), 0.0);

        let (mut multi_level, _) = build_adaptive(mat.clone(), cf, 10, 0.01, 10, &prefix);
        let mut x: DVector<f64> = rand.clone();
        let (converged_multi, ratio_multi, iters_multi) =
            stationary(&mat, &b, &mut x, 10000, epsilon, &multi_level, Some(3));
        if !converged_multi {
            warn!("multi level didn't converge with ratio: {}", ratio_multi);
        }
        let levels = multi_level.components()[0].get_hierarchy().levels();
        /*

        let pc = &mut multi_level.components_mut()[0];
        while pc.hierarchy.levels() > 2 {
            pc.hierarchy.matrices.pop();
        }
        //let (two_level, _) = build_adaptive(mat.clone(), cf, 2, 0.01, 1, &prefix);
        let mut x: DVector<f64> = rand.clone();
        let (converged_2, ratio_2, iters_2) =
            stationary(&mat, &b, &mut x, 10000, epsilon, &multi_level, Some(3));
        if !converged_2 {
            warn!("2 level didn't converge with ratio: {}", ratio_2);
        }
        */

        results.push((dim, 0, iters_multi, levels));

        println!(
            "{:>15} {:>15} {:>15} {:>15}",
            "size", "2-level", "multi-level", "levels"
        );
        for (dim, two, multi, levels) in results.iter() {
            println!("{:15} {:15} {:15} {:15}", dim, two, multi, levels);
        }
    }
}
