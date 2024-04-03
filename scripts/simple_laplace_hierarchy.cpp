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
  const char *mesh_file = "../meshes/star.mesh";
  int order = 1;
  int refinements = 8;

  Mesh mesh(mesh_file);
  for (int i = 0; i < refinements; ++i) {

    H1_FECollection fec(order, mesh.Dimension());
    FiniteElementSpace fespace(&mesh, &fec);
    cout << "Refinement number: " << i << endl;
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
    //a.AddDomainIntegrator(new MassIntegrator);
    a.Assemble();

    SparseMatrix A;
    Vector B, X;
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
    cout << "Size of linear system: " << A.Height() << endl;

    std::ofstream matfile(std::format("../laplace/{}.mtx", i), std::ios::out);
    A.PrintMM(matfile);
    matfile.close();

    std::ofstream vecfile(std::format("../laplace/{}.rhs", i), std::ios::out);
    B.Print(vecfile);
    vecfile.close();

    std::ofstream bdyfile(std::format("../laplace/{}.bdy", i), std::ios::out);
    boundary_dofs.Save(bdyfile);
    bdyfile.close();

    mesh.UniformRefinement();
  }
  return 0;
}
