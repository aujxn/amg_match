#include "spe10.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  int order = 1;
  int refinements = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&refinements, "-r", "--refinements",
                 "How many times to uniform refine");
  args.ParseCheck();

  Mesh mesh =
      Mesh::MakeCartesian3D(60, 220, 85, Element::HEXAHEDRON, 1200, 2200, 170);

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

  SPE10Coefficient coef;
  BilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator(coef));
  a.Assemble();

  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
  cout << "Size of linear system: " << A.Height() << endl;

  std::ofstream matfile("data/spe10/spe10_0.mtx", std::ios::out);
  A.PrintMM(matfile);
  matfile.close();

  std::ofstream vecfile("data/spe10/spe10_0.rhs", std::ios::out);
  B.Print(vecfile);
  vecfile.close();

  std::ofstream bdyfile("data/spe10/spe10_0.bdy", std::ios::out);
  boundary_dofs.Save(bdyfile);
  bdyfile.close();
  // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
  // GSSmoother M(A);
  // PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

  // 10. Recover the solution x as a grid function and save to file. The output
  //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
  // a.RecoverFEMSolution(X, b, x);
  // x.Save("sol.gf");
  // mesh.Save("mesh.mesh");

  return 0;
}
