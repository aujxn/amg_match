#include "mfem.hpp"
#include <format>
#include <fstream>
#include <iostream>
#include <mfem/general/tic_toc.hpp>
#include <mfem/linalg/solvers.hpp>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const char *mesh_file = "../data/meshes/star.mesh";
  int order = 1;
  int refinements = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&refinements, "-r", "--refinements",
                 "How many times to uniform refine");
  args.ParseCheck();

  Mesh serial_mesh(mesh_file);
  ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear(); // the serial mesh is no longer needed
  int dim = mesh.Dimension();
  for (int i = 0; i < refinements; ++i) {
    mesh.UniformRefinement();
  }

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fespace(&mesh, &fec);
  HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();
  if (Mpi::Root()) {
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

  // double epsilon = 1e-6;
  double epsilon = 1e-4;
  double theta = (60.0 / 180.0) * 3.1415;
  double phi = (27.0 / 180.0) * 3.1415;

  MatrixConstantCoefficient *diffusion_coef;

  if (dim == 3) {
    DenseMatrix angle(3, 1);
    angle.Elem(0, 0) = cos(theta) * cos(phi);
    angle.Elem(1, 0) = sin(theta) * cos(phi);
    angle.Elem(2, 0) = sin(phi);
    DenseMatrix coef(3, 3);
    MultAAt(angle, coef);
    coef.Elem(0, 0) += epsilon;
    coef.Elem(1, 1) += epsilon;
    coef.Elem(2, 2) += epsilon;
    // cout << "Coefficient matrix: ";
    // coef.Print();

    diffusion_coef = new MatrixConstantCoefficient(coef);
  } else if (dim == 2) {
    DenseMatrix angle(2, 1);
    angle.Elem(0, 0) = cos(theta);
    angle.Elem(1, 0) = sin(theta);
    DenseMatrix coef(2, 2);
    MultAAt(angle, coef);
    coef.Elem(0, 0) += epsilon;
    coef.Elem(1, 1) += epsilon;
    // cout << "Coefficient matrix: ";
    // coef.Print();

    diffusion_coef = new MatrixConstantCoefficient(coef);
  } else {
    cout << "Not a 2 or 3 dimensional mesh?";
    return 1;
  }

  ParBilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator(*diffusion_coef));
  a.Assemble();

  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

  StopWatch timer;
  if (Mpi::Root()) {
    cout << "Size of linear system: " << A.Height() << endl;
    timer.Start();
  }

  HypreBoomerAMG M(A);

  MPI_Barrier(MPI_COMM_WORLD);
  if (Mpi::Root()) {
    cout << "Preconditioner constructed in: " << timer.RealTime() << endl;
    ;
    timer.Clear();
    timer.Start();
  }

  // CGSolver solver(MPI_COMM_WORLD);
  SLISolver solver(MPI_COMM_WORLD);
  M.SetRelaxType(18);
  M.SetCycleNumSweeps(3, 3);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  mfem::IterativeSolver::PrintLevel pl;
  solver.SetPrintLevel(pl.Summary());
  solver.SetOperator(A);
  solver.SetPreconditioner(M);
  solver.Mult(B, X);

  MPI_Barrier(MPI_COMM_WORLD);

  if (Mpi::Root()) {
    cout << "final relative norm: " << solver.GetFinalRelNorm() << endl
         << " Solve time: " << timer.RealTime() << endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}
