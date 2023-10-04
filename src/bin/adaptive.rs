use amg_match::{
    adaptive::build_adaptive,
    preconditioner::Composite,
    solver::pcg,
    utils::{delete_boundary, format_duration, load_boundary_dofs, load_vec, random_vec},
};
use nalgebra_sparse::{io::load_coo_from_matrix_market_file as load_mm, CsrMatrix};
use plotly::{
    layout::{Annotation, Axis, AxisType},
    Layout, Plot, Scatter,
};
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

    /// Desired convergence rate per iteration
    #[structopt()]
    target_convergence: f64,

    /// Max number of components in preconditioner
    #[structopt()]
    max_components: usize,

    /// File/Path to save the serialized preconditioner to
    #[structopt(parse(from_os_str))]
    output: std::path::PathBuf,
}

fn main() {
    pretty_env_logger::init();
    let opt = Opt::from_args();

    let matfile = format!("{}.mtx", &opt.input);
    let doffile = format!("{}.bdy", &opt.input);
    let rhsfile = format!("{}.rhs", &opt.input);

    let (mat, b) = {
        let mat = CsrMatrix::from(&load_mm(matfile).unwrap());

        let b = load_vec(rhsfile);
        let dofs = load_boundary_dofs(doffile);

        let (mat, b) = delete_boundary(dofs, mat, b);
        (std::rc::Rc::new(mat), b)
    };

    let timer = std::time::Instant::now();
    let (pc, test_data) = build_adaptive(
        mat.clone(),
        opt.coarsening_factor,
        opt.max_levels,
        opt.target_convergence,
        opt.max_components,
    );

    info!(
        "Preconitioner built in: {}",
        format_duration(timer.elapsed())
    );
    let notes = format!("{:?}", &opt);
    let title = format!(
        "PC Test: input: {}, coarsening factor: {}, max levels: {}",
        opt.input, opt.coarsening_factor, opt.max_levels
    );
    plot(&test_data, &title);
    pc.save(opt.output, notes);

    //let (pc, notes) = Composite::load(mat.clone(), "data/out/test_pc.json");
    //println!("notes: {}", notes);
    // TODO add plot data to the PC serialization

    let timer = std::time::Instant::now();
    let mut x = random_vec(mat.nrows());
    let _ = pcg(&mat, &b, &mut x, 1000, 1e-6, &pc, Some(3));
    info!("Solved in: {}", format_duration(timer.elapsed()));
}

fn plot(data: &Vec<Vec<f64>>, title: &String) {
    let mut plot = Plot::new();
    for trace in data.iter() {
        let trace = Scatter::new((1..=trace.len()).collect(), trace.clone());
        plot.add_trace(trace);
    }
    //let date = format!("{:?}", chrono::offset::Local::now());
    //let filename = format!("data/out/{}.html", &title);
    let layout = Layout::new()
        .title(title.as_str().into())
        .y_axis(
            Axis::new()
                .title("Relative Error in A-norm".into())
                .type_(AxisType::Log),
        )
        .x_axis(Axis::new().title("Iteration".into()));

    plot.set_layout(layout);
    plot.show();
    //plot.write_html(&filename);
}
