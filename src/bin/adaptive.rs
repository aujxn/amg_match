use amg_match::{
    adaptive::build_adaptive,
    utils::{delete_boundary, load_boundary_dofs, load_vec},
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
    #[structopt(parse(from_os_str))]
    input: std::path::PathBuf,

    #[structopt()]
    coarsening_factor: f64,

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

    let mat = {
        let mat = CsrMatrix::from(
            &nalgebra_sparse::io::load_coo_from_matrix_market_file(&opt.input).unwrap(),
        );

        let b = load_vec("data/spe10/spe10_0.rhs");
        let dofs = load_boundary_dofs("data/spe10/spe10_0.bdy");

        let (mat, _b) = delete_boundary(dofs, mat, b);

        std::rc::Rc::new(mat)
    };

    let timer = std::time::Instant::now();
    let pc = Box::new(build_adaptive(mat.clone(), opt.coarsening_factor));

    info!(
        "Preconitioner built in: {} seconds.",
        timer.elapsed().as_secs()
    );

    pc.save(opt.output, "spe10, no refinements, 20 components".into());
    /*
    let (preconditioner, notes) = Composite::load(mat.clone(), "data/out/test_pc.json");
    println!("notes: {}", notes);
    */
}
