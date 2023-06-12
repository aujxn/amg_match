use amg_match::{
    adaptive::build_adaptive,
    preconditioner::Composite,
    solver::pcg,
    utils::{delete_boundary, format_duration, load_boundary_dofs, load_vec, random_vec},
};
use nalgebra_sparse::CsrMatrix;
use structopt::StructOpt;

#[macro_use]
extern crate log;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "amg_match_cli",
    about = "CLI for testing different iterative solvers and preconditioners"
)]
struct Opt {
    /// Matrix file in matrix market format
    #[structopt()]
    input: String,

    /// Coarsening factor per level in each AMG hierarchy
    #[structopt()]
    coarsening_factor: f64,

    /// Max levels in each hierarchy
    #[structopt()]
    max_levels: usize,

    /// File/Path to save the serialized preconditioner to
    #[structopt(parse(from_os_str))]
    output: std::path::PathBuf,
    /*
    #[structopt(default_value = "8" )]
    components: usize,
    */
}
fn main() {
    pretty_env_logger::init();
    let opt = Opt::from_args();

    let (mat, b) = {
        let mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(format!("{}.mtx", &opt.input))
                .unwrap(),
        );

        let b = load_vec(format!("{}.rhs", &opt.input));
        let dofs = load_boundary_dofs(format!("{}.bdy", &opt.input));

        let (mat, b) = delete_boundary(dofs, mat, b);

        (std::rc::Rc::new(mat), b)
    };

    let timer = std::time::Instant::now();
    let pc = build_adaptive(mat.clone(), opt.coarsening_factor, opt.max_levels);

    info!(
        "Preconitioner built in: {}",
        format_duration(timer.elapsed())
    );

    pc.save(opt.output, "spe10, no refinements, 20 components".into());

    //let (pc, notes) = Composite::load(mat.clone(), "data/out/test_pc.json");
    //println!("notes: {}", notes);
    // TODO add plot data to the PC serialization

    let timer = std::time::Instant::now();
    let mut x = random_vec(mat.nrows());
    let _ = pcg(&mat, &b, &mut x, 1000, 1e-6, &pc, Some(3));
    info!("Solved in: {}", format_duration(timer.elapsed()));
}
