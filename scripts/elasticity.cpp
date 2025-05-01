//                                MFEM Example 2
// Modified code from mfem/examples/ex2.cpp
//
// Example run:  ex2 -m ../data/beam-tri.mesh -o 1 -r 4
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <mfem/fem/fespace.hpp>

using namespace std;
using namespace mfem;
int generate_rbms(FiniteElementSpace *fespace, char *folder);

int main(int argc, char *argv[]) {
  const char *mesh_file = "../data/beam-tet.mesh";
  int order = 1;
  bool visualization = 1;
  int refinements = 2;
  auto ordering = Ordering::byVDIM;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&refinements, "-r", "--refinements",
                 "How many times to uniform refine");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  if (mesh->NURBSext) {
    cerr << "\nNo NURBS please\n";
    return 3;
  }

  if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2) {
    cerr << "\nInput mesh should have at least two materials and "
         << "two boundary attributes! (See schematic in ex2.cpp)\n"
         << endl;
    return 3;
  }

  {
    for (int i = 0; i < refinements; ++i) {
      mesh->UniformRefinement();
    }
  }

  FiniteElementCollection *fec;
  FiniteElementSpace *fespace;
  fec = new H1_FECollection(order, dim);
  fespace = new FiniteElementSpace(mesh, fec, dim, ordering);
  cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
       << endl
       << "Assembling: " << flush;

  char filename[200];
  const char *base = "../data/elasticity";
  sprintf(filename, "%s/%d", base, refinements);
  generate_rbms(fespace, filename);
  Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  VectorArrayCoefficient f(dim);
  for (int i = 0; i < dim - 1; i++) {
    f.Set(i, new ConstantCoefficient(0.0));
  }
  {
    Vector pull_force(mesh->bdr_attributes.Max());
    pull_force = 0.0;
    pull_force(1) = -1.0e-2;
    f.Set(dim - 1, new PWConstCoefficient(pull_force));
  }

  LinearForm *b = new LinearForm(fespace);
  b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
  cout << "r.h.s. ... " << flush;
  b->Assemble();

  GridFunction x(fespace);
  x = 0.0;

  Vector lambda(mesh->attributes.Max());
  lambda = 1.0;
  lambda(0) = lambda(1) * 50;
  PWConstCoefficient lambda_func(lambda);
  Vector mu(mesh->attributes.Max());
  mu = 1.0;
  mu(0) = mu(1) * 50;
  PWConstCoefficient mu_func(mu);

  BilinearForm *a = new BilinearForm(fespace);
  a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

  cout << "matrix ... " << flush;
  a->Assemble();

  SparseMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
  cout << "done." << endl;

  cout << "Size of linear system: " << A.Height() << endl;

  sprintf(filename, "%s/%d/elasticity_3d.mtx", base, refinements);
  std::ofstream matfile(filename, std::ios::out);
  A.PrintMM(matfile);
  matfile.close();

  sprintf(filename, "%s/%d/elasticity_3d.rhs", base, refinements);
  std::ofstream vecfile(filename, std::ios::out);
  B.Print(vecfile);
  vecfile.close();

  sprintf(filename, "%s/%d/elasticity_3d.bdy", base, refinements);
  std::ofstream bdyfile(filename, std::ios::out);
  ess_tdof_list.Save(bdyfile);
  bdyfile.close();

#ifndef MFEM_USE_SUITESPARSE
  GSSmoother M(A);
  PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
#else
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(A);
  umf_solver.Mult(B, X);
#endif

  a->RecoverFEMSolution(X, *b, x);
  mesh->SetNodalFESpace(fespace);

  delete a;
  delete b;
  if (fec) {
    delete fespace;
    delete fec;
  }
  delete mesh;

  return 0;
}

// Rotational rigid-body mode functions, used in SetElasticityOptions()
static void func_rxy(const Vector &x, Vector &y) {
  y = 0.0;
  y(0) = x(1);
  y(1) = -x(0);
}
static void func_ryz(const Vector &x, Vector &y) {
  y = 0.0;
  y(1) = x(2);
  y(2) = -x(1);
}
static void func_rzx(const Vector &x, Vector &y) {
  y = 0.0;
  y(2) = x(0);
  y(0) = -x(2);
}

// Translational rigid-body mode functions
static void func_tx(const Vector &x, Vector &y) {
  y = 0.0;
  y(0) = 1.0;
}
static void func_ty(const Vector &x, Vector &y) {
  y = 0.0;
  y(1) = 1.0;
}
static void func_tz(const Vector &x, Vector &y) {
  y = 0.0;
  y(2) = 1.0;
}

int generate_rbms(FiniteElementSpace *fespace, char *folder) {

  // translational modes, taken from mfem/linalg/hypre.cpp line 5262
  VectorFunctionCoefficient coeff_rxy(3, func_rxy);
  VectorFunctionCoefficient coeff_ryz(3, func_ryz);
  VectorFunctionCoefficient coeff_rzx(3, func_rzx);
  GridFunction rbms_rxy(fespace);
  GridFunction rbms_ryz(fespace);
  GridFunction rbms_rzx(fespace);
  rbms_rxy.ProjectCoefficient(coeff_rxy);
  rbms_ryz.ProjectCoefficient(coeff_ryz);
  rbms_rzx.ProjectCoefficient(coeff_rzx);

  char filename[200];
  sprintf(filename, "%s/%s", folder, "rbm_rotate_xy.gf");
  rbms_rxy.Save(filename);
  sprintf(filename, "%s/%s", folder, "rbm_rotate_yz.gf");
  rbms_ryz.Save(filename);
  sprintf(filename, "%s/%s", folder, "rbm_rotate_zx.gf");
  rbms_rzx.Save(filename);

  // rotational modes
  VectorFunctionCoefficient coeff_tx(3, func_tx);
  VectorFunctionCoefficient coeff_ty(3, func_ty);
  VectorFunctionCoefficient coeff_tz(3, func_tz);
  GridFunction rbms_tx(fespace);
  GridFunction rbms_ty(fespace);
  GridFunction rbms_tz(fespace);
  rbms_tx.ProjectCoefficient(coeff_tx);
  rbms_ty.ProjectCoefficient(coeff_ty);
  rbms_tz.ProjectCoefficient(coeff_tz);

  sprintf(filename, "%s/%s", folder, "rbm_translate_x.gf");
  rbms_tx.Save(filename);
  sprintf(filename, "%s/%s", folder, "rbm_translate_y.gf");
  rbms_ty.Save(filename);
  sprintf(filename, "%s/%s", folder, "rbm_translate_z.gf");
  rbms_tz.Save(filename);
  return 0;
}
