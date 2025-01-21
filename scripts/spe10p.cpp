#include "mfem.hpp"
#include "spe10.hpp"
#include <format>
#include <fstream>
#include <iostream>
#include <mfem/linalg/operator.hpp>
#include <mfem/linalg/solvers.hpp>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  int order = 1;
  int refinements = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&refinements, "-r", "--refinements",
                 "How many times to uniform refine");
  args.ParseCheck();

  int x_dim = 60;
  int y_dim = 220;
  int z_dim = 85;
  /*
  int x_dim = 3;
  int y_dim = 10;
  int z_dim = 50;
  */

  // Mesh mesh = Mesh::MakeCartesian3D(x_dim, y_dim, z_dim, Element::HEXAHEDRON,
  // 1200, 2200, 170);
  Mesh serial_mesh = Mesh::MakeCartesian3D(
      // 60, 220, 85, Element::HEXAHEDRON, x_dim, y_dim, z_dim);
      // 60, 220, 85, Element::HEXAHEDRON);
      x_dim, y_dim, z_dim, Element::HEXAHEDRON, 1200, 2200, 170);
  ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear(); // the serial mesh is no longer needed

  for (int i = 0; i < refinements; ++i) {
    mesh.UniformRefinement();
  }

  H1_FECollection fec(order, mesh.Dimension());
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

  SPE10Coefficient coef;
  // ScalarSPE10Coefficient coef;
  ParBilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator(coef));
  // a.AddDomainIntegrator(new DiffusionIntegrator);
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
  // M.SetRelaxType(8);
  // M.SetCycleType(2);
  CGSolver solver(MPI_COMM_WORLD);
  // SLISolver solver(MPI_COMM_WORLD);
  M.SetRelaxType(18);
  M.SetCycleNumSweeps(3, 3);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);
  solver.SetOperator(A);
  solver.SetPreconditioner(M);
  solver.Mult(B, X);

  if (Mpi::Root()) {
    cout << "final relative norm: " << solver.GetFinalRelNorm() << endl
         << " Solve time: " << timer.RealTime() << endl;
  }
  // 12. Recover the solution x as a grid function and save to file. The output
  //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
  a.RecoverFEMSolution(X, b, x);
  x.Save("sol");
  mesh.Save("mesh");
  /*
  A.EliminateBC(boundary_dofs, mfem::Operator::DIAG_ZERO);
  A.Threshold(1e-12);
  cout << "Size of linear system: " << A.Height() << endl;
  cout << "Linear system nnz: " << A.NumNonZeroElems() << endl;

  std::ofstream matfile("../spe10/spe10_0.mtx", std::ios::out);
  A.PrintMM(matfile);
  matfile.close();

  std::ofstream vecfile("../spe10/spe10_0.rhs", std::ios::out);
  B.Print(vecfile);
  vecfile.close();

  std::ofstream bdyfile("../spe10/spe10_0.bdy", std::ios::out);
  boundary_dofs.Save(bdyfile);
  bdyfile.close();

  cout << "Building Scalar Coef..." << endl;
  ScalarSPE10Coefficient scalar_coef;
  DG_FECollection fec_dg(0, 3);
  FiniteElementSpace fespace_dg(&mesh, &fec_dg);
  GridFunction coef_gf(&fespace_dg);
  cout << "Projecting..." << endl;
  coef_gf.ProjectCoefficient(scalar_coef);

  cout << "Saving coefficient..." << endl;
  std::ofstream coeffile("../spe10/spe10_coef.vtk", std::ios::out);
  coeffile.precision(14);
  mesh.PrintVTK(coeffile, 1);
  coef_gf.SaveVTK(coeffile, "permiability", 1);

  int dim = mesh.Dimension();
  std::ofstream coordsfile(std::format("../spe10/spe10_0.coords", dim),
                           std::ios::out);

  FiniteElementSpace fes2(&mesh, &fec, mesh.SpaceDimension());
  GridFunction nodes(&fes2);
  mesh.GetNodes(nodes);
  // nodes.Save(coordsfile);

  const int nNodes = nodes.Size() / dim;
  double coord[dim]; // coordinates of a node
  for (int i = 0; i < nNodes; ++i) {
    for (int j = 0; j < dim; ++j) {
      coord[j] = nodes(j * nNodes + i);
      coordsfile << coord[j] << " ";
    }
    coordsfile << endl;
  }
  // 9. Solve the system using Psolver with symmetric Gauss-Seidel
  // preconditioner. GSSmoother M(A); Psolver(A, M, B, X, 1, 200, 1e-12, 0.0);

  // 10. Recover the solution x as a grid function and save to file. The
  // output
  //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g
  //     sol.gf"
  // a.RecoverFEMSolution(X, b, x);
  // x.Save("sol.gf");
  // mesh.Save("mesh.mesh");
  // */

  return 0;
}
