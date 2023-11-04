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

#[macro_use]
extern crate log;

pub mod adaptive;
pub mod io;
//pub mod parallel_ops;
pub mod partitioner;
pub mod preconditioner;
pub mod solver;
pub mod suitesparse_dl;
pub mod utils;
