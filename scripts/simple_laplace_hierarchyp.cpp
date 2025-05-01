
//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <format>
#include <fstream>
#include <iostream>
#include <mfem/fem/bilininteg.hpp>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const char *mesh_file = "../data/meshes/beam-tri.vtk";
  int order = 1;
  int refinements = 10;

  Mesh serial_mesh(mesh_file);
  ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear(); // the serial mesh is no longer needed
  for (int i = 0; i < refinements; ++i) {

    H1_FECollection fec(order, mesh.Dimension());
    ParFiniteElementSpace fespace(&mesh, &fec);
    HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();
    if (Mpi::Root()) {
      cout << "Refinement number: " << i << endl;
      cout << "Number of unknowns: " << total_num_dofs << endl;
    }

    Array<int> boundary_dofs;
    fespace.GetBoundaryTrueDofs(boundary_dofs);

    ParGridFunction x(&fespace);
    x = 0.0;

    ConstantCoefficient one(1.0);
    ParLinearForm b(&fespace);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));
    b.Assemble();

    ParBilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator);
    // a.AddDomainIntegrator(new MassIntegrator);
    a.Assemble();

    HypreParMatrix A;
    Vector B, X;
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
    if (Mpi::Root()) {
      cout << "Size of linear system: " << A.Height() << endl;
    }

    HypreBoomerAMG M(A);
    M.SetInterpolation(0);

    // CGSolver solver(MPI_COMM_WORLD);
    SLISolver solver(MPI_COMM_WORLD);
    // M.SetCycleNumSweeps(3, 3);
    solver.SetRelTol(1e-8);
    solver.SetMaxIter(10000);
    mfem::IterativeSolver::PrintLevel pl;
    solver.SetPrintLevel(pl.Summary());
    solver.SetOperator(A);
    solver.SetPreconditioner(M);
    solver.Mult(B, X);

    mesh.UniformRefinement();
  }
  return 0;
}
