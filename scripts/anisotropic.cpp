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
  const char *mesh_file = "../data/star.mesh";
  int order = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.ParseCheck();

  Mesh mesh(mesh_file);
  mesh.UniformRefinement();

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

  BilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator);
  a.Assemble();

  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
  cout << "Size of linear system: " << A.Height() << endl;

  /*
  GSSmoother M(A);
  PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

  a.RecoverFEMSolution(X, b, x);
  x.Save("sol.gf");
  mesh.Save("mesh.mesh");
  */

  return 0;
}
