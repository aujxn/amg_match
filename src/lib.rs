//! TODO add links to github, crates, and docs
//!
//! <br>
//!
//! This library provides a testing suite for the research algorithms described in
//! the paper: TODO add link to paper.
//!
//! The solvers implemented are intended for symmetric positive definite matrices.
//! These matrices often arise from discretizations of elliptic operators in
//! partial differential equations describing diffusion.
//! The optimal and popular preconditioners for these discretizations are multigrid
//! methods. For fairly regular meshes and material compositions, standard
//! geometric multigrid is simple and performant. In the case of more complex
//! geometries and materials, especially those exhibiting extreme anisotropies,
//! require more generalized and sophisticated algebraic multigrid (AMG) techniques.
//! Many variants and approaches to AMG exists, many of which are quite complicated
//! and specialized. This implementation adds a general and (relatively) simple option
//! to those methods.
//!
//! The algorithms implemented are based on construction of composite algebraic
//! multigrid preconditioners. Each component of the composite preconditioner
//! is adaptive and utilizes vectors from the 'near null' space of the matrix
//! in the system. It is well known that these 'near null' components, or
//! low magnitude eigenmodes, are difficult to eliminate with Krylov subspace
//! based iterative methods. These 'near null' components are discovered by
//! testing the preconditioner on the system `Ax=0` until convergence stalls.
//! Afterwords, they are used to construct a specialized AMG cycle which is
//! composed with the previous preconditioner to form a new composite method.
//! This cycle of testing, constructing, and composing is repeated until the
//! method has a desired rate of convergence on the test problem.

use ndarray::{Array1, Array2};
use sprs::{CsMatBase, TriMatBase};
use sprs_ldl::LdlNumeric;
use num_cpus;

#[macro_use]
extern crate log;
extern crate approx;

pub mod adaptive;
pub mod hierarchy;
pub mod interpolation;
pub mod parallel_ops;
pub mod partitioner;
pub mod preconditioner;
pub mod solver;
pub mod utils;

pub type CsrMatrix = CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>, usize>;
pub type CooMatrix = TriMatBase<Vec<usize>, Vec<f64>>;
pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;
pub type Cholesky = LdlNumeric<f64, usize>;

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Local;
use lazy_static::lazy_static;

// Lazily-initialised output directory.
lazy_static! {
    static ref OUTPUT_DIR: PathBuf = {

        let slurm_job_id = std::env::var("SLURM_JOB_ID");
        match slurm_job_id {
            Ok(id) => {
                // if we are on the orca cluster save to scratch
                let base = Path::new("/scratch/ajn6-amg");
                let slurm_job_name = std::env::var("SLURM_JOB_NAME").unwrap();
                return base.join(format!("{}-{}", slurm_job_name, id));
            },
            Err(_) => {
                let base = Path::new("./output");
                fs::create_dir_all(base)
                    .expect("Failed to create base output directory");
                let ts = Local::now().format("%Y-%m-%d_%H:%M:%S").to_string();
                for suffix in 0u32.. {
                    let candidate = if suffix == 0 {
                        base.join(&ts)
                    } else {
                        base.join(format!("{}_{}", ts, suffix))
                    };

                    if !candidate.exists() {
                        fs::create_dir_all(&candidate)
                            .expect("Failed to create unique output directory");
                        return candidate;
                    }
                }
                unreachable!("u32 exhausted while searching for unique directory name")
            }
        }
    };
    static ref N_CPUS: usize = num_cpus::get();
}

/// Helper to build paths inside the output directory.
///
/// ```
/// use std::io::Write;
///
/// let path = output_path("example.txt");
/// let mut f = fs::File::create(&path)?;
/// writeln!(f, "Hello, world!")?;
///
/// println!("Wrote to {}", path.display());
/// ```
pub fn output_path<S: AsRef<Path>>(file: S) -> PathBuf {
    OUTPUT_DIR.join(file)
}
