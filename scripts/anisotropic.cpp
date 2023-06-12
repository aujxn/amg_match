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
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  const char *mesh_file = "data/meshes/star.mesh";
  int order = 1;
  int refinements = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&refinements, "-r", "--refinements",
                 "How many times to uniform refine");
  args.ParseCheck();

  Mesh mesh(mesh_file);
  for (int i = 0; i < refinements; ++i) {
    mesh.UniformRefinement();
  }

  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

  Array<int> boundary_dofs;
  fespace.GetBoundaryTrueDofs(boundary_dofs);

  GridFunction x(&fespace);
  x = 0.0;

  ConstantCoefficient one(1.0);
  LinearForm b(&fespace);
  b.AddDomainIntegrator(new DomainLFIntegrator(one));
  b.Assemble();

  double epsilon = 0.001;
  double theta = (55.0 / 180.0) * 3.1415;
  double phi = (27.0 / 180.0) * 3.1415;

  DenseMatrix angle(3, 1);
  angle.Elem(0, 0) = cos(theta) * cos(phi);
  angle.Elem(1, 0) = sin(theta) * cos(phi);
  angle.Elem(2, 0) = sin(phi);
  DenseMatrix coef(3, 3);
  MultAAt(angle, coef);
  coef.Elem(0, 0) += epsilon;
  coef.Elem(1, 1) += epsilon;
  coef.Elem(2, 2) += epsilon;
  cout << "Coefficient matrix: ";
  coef.Print();

  MatrixConstantCoefficient diffusion_coef(coef);

  BilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coef));
  a.Assemble();

  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
  cout << "Size of linear system: " << A.Height() << endl;

  std::ofstream matfile("data/anisotropy/anisotropy_3d.mtx", std::ios::out);
  A.PrintMM(matfile);
  matfile.close();

  std::ofstream vecfile("data/anisotropy/anisotropy_3d.rhs", std::ios::out);
  B.Print(vecfile);
  vecfile.close();

  std::ofstream bdyfile("data/anisotropy/anisotropy_3d.bdy", std::ios::out);
  boundary_dofs.Save(bdyfile);
  bdyfile.close();
  /*
  GSSmoother M(A);
  PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

  a.RecoverFEMSolution(X, b, x);
  x.Save("sol.gf");
  mesh.Save("mesh.mesh");
  */

  return 0;
}
