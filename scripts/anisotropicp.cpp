#include "mfem.hpp"
#include <format>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const char *mesh_file = "../meshes/star.mesh";
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

  double epsilon = 1e-6;
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
  if (Mpi::Root()) {
    cout << "Size of linear system: " << A.Height() << endl;
  }

  HypreBoomerAMG M(A);

  // CGSolver cg(MPI_COMM_WORLD);
  SLISolver cg(MPI_COMM_WORLD);
  M.SetRelaxType(18);
  M.SetCycleNumSweeps(15, 15);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(10000);
  cg.SetPrintLevel(1);
  cg.SetOperator(A);
  cg.SetPreconditioner(M);
  cg.Mult(B, X);

  if (Mpi::Root()) {
    cout << "final relative norm: " << cg.GetFinalRelNorm();
  }

  return 0;
}
