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

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  const char *mesh_file = "../meshes/star.mesh";
  int order = 1;
  int refinements = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&refinements, "-r", "--refinements",
                 "How many times to uniform refine");
  args.ParseCheck();

  Mesh mesh(mesh_file);
  int dim = mesh.Dimension();
  for (int i = 0; i < refinements; ++i) {
    mesh.UniformRefinement();
  }

  H1_FECollection fec(order, dim);
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
    cout << "Coefficient matrix: ";
    coef.Print();

    diffusion_coef = new MatrixConstantCoefficient(coef);
  } else if (dim == 2) {
    DenseMatrix angle(2, 1);
    angle.Elem(0, 0) = cos(theta);
    angle.Elem(1, 0) = sin(theta);
    DenseMatrix coef(2, 2);
    MultAAt(angle, coef);
    coef.Elem(0, 0) += epsilon;
    coef.Elem(1, 1) += epsilon;
    cout << "Coefficient matrix: ";
    coef.Print();

    diffusion_coef = new MatrixConstantCoefficient(coef);
  } else {
    cout << "Not a 2 or 3 dimensional mesh?";
    return 1;
  }

  BilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator(*diffusion_coef));
  a.Assemble();

  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
  cout << "Size of linear system: " << A.Height() << endl;

  std::ofstream matfile(std::format("../anisotropy/anisotropy_{}d.mtx", dim),
                        std::ios::out);
  A.PrintMM(matfile);
  matfile.close();

  std::ofstream vecfile(std::format("../anisotropy/anisotropy_{}d.rhs", dim),
                        std::ios::out);
  B.Print(vecfile);
  vecfile.close();

  std::ofstream bdyfile(std::format("../anisotropy/anisotropy_{}d.bdy", dim),
                        std::ios::out);
  boundary_dofs.Save(bdyfile);
  bdyfile.close();

  std::ofstream coordsfile(
      std::format("../anisotropy/anisotropy_{}d.coords", dim), std::ios::out);

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

  ofstream omesh("../anisotropy/test.vtk");
  omesh.precision(14);
  mesh.PrintVTK(omesh);
  cout << "New VTK mesh file: " << mesh_file << endl;
  /*
  GridFunction W_coords(&fespace);
  {
    DenseMatrix coords, coords_t;
    Array<int> wt_vdofs;
    for (int i = 0; i < mesh.GetNE(); i++) {
      const FiniteElement *wt_fe = fespace.GetFE(i);
      const IntegrationRule &wt_nodes = wt_fe->GetNodes();
      ElementTransformation *T = mesh.GetElementTransformation(i);
      T->Transform(wt_nodes, coords);
      coords_t.Transpose(coords);
      fespace.GetElementVDofs(i, wt_vdofs);
      W_coords.SetSubVector(wt_vdofs, coords_t.GetData());
    }
  }

  coordsfile << W_coords;
  coordsfile.close();
  */

  /*
  GSSmoother M(A);
  PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

  a.RecoverFEMSolution(X, b, x);
  x.Save("sol.gf");
  mesh.Save("mesh.mesh");
  */

  return 0;
}
