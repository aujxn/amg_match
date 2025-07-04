#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <mfem/linalg/solvers.hpp>
#include <ostream>

using namespace std;
using namespace mfem;

// mpiexec -n 8 ./elasticityp -m ../data/meshes/beam-tet.mesh -r 4
int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  const char *mesh_file = "../data/meshes/beam-tet.mesh";
  const char *solver_method = "SLI";
  int order = 1;
  bool amg_elast = 0;

  // boomerAMG depends on array-of-structures (AoS) ordering which is byVDIM
  auto ordering = Ordering::byVDIM;
  // auto ordering = Ordering::byNODES; // structure-of-arrays SoA

  int refinements = 2;
  const char *device_config = "cpu";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                 "--amg-for-systems",
                 "Use the special AMG elasticity solver (GM/LN approaches), "
                 "or standard AMG for systems (unknown approach).");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&refinements, "-r", "--refinements",
                 "How many times to uniform refine");
  args.AddOption(&solver_method, "-s", "--solver", "Either CG or SLI");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  //    Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(device_config);
  if (myid == 0) {
    device.Print();
  }

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2) {
    if (myid == 0)
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
    return 3;
  }

  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;
  {
    for (int l = 0; l < refinements; l++) {
      pmesh->UniformRefinement();
    }
  }

  //    Define a parallel finite element space on the parallel mesh. Here we
  //    use vector finite elements, i.e. dim copies of a scalar finite element
  //    space. We use the ordering by vector dimension (the last argument of
  //    the FiniteElementSpace constructor) which is expected in the systems
  //    version of BoomerAMG preconditioner.
  FiniteElementCollection *fec;
  ParFiniteElementSpace *fespace;
  fec = new H1_FECollection(order, dim);
  fespace = new ParFiniteElementSpace(pmesh, fec, dim, ordering);

  HYPRE_BigInt size = fespace->GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl
         << "Assembling: " << flush;
  }

  Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  //     Set up the parallel linear form b(.) which corresponds to the
  //     right-hand side of the FEM linear system. In this case, b_i equals the
  //     boundary integral of f*phi_i where f represents a "pull down" force on
  //     the Neumann part of the boundary and phi_i are the basis functions in
  //     the finite element fespace. The force is defined by the object f, which
  //     is a vector of Coefficient objects. The fact that f is non-zero on
  //     boundary attribute 2 is indicated by the use of piece-wise constants
  //     coefficient for its last component.
  VectorArrayCoefficient f(dim);
  for (int i = 0; i < dim - 1; i++) {
    f.Set(i, new ConstantCoefficient(0.0));
  }
  {
    Vector pull_force(pmesh->bdr_attributes.Max());
    pull_force = 0.0;
    pull_force(1) = -1.0e-2;
    f.Set(dim - 1, new PWConstCoefficient(pull_force));
  }

  ParLinearForm *b = new ParLinearForm(fespace);
  b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
  if (myid == 0) {
    cout << "r.h.s. ... " << flush;
  }
  b->Assemble();

  ParGridFunction x(fespace);
  x = 0.0;

  //     Set up the parallel bilinear form a(.,.) on the finite element space
  //     corresponding to the linear elasticity integrator with piece-wise
  //     constants coefficient lambda and mu.
  Vector lambda(pmesh->attributes.Max());
  lambda = 1.0;
  lambda(0) = lambda(1) * 50;
  PWConstCoefficient lambda_func(lambda);
  Vector mu(pmesh->attributes.Max());
  mu = 1.0;
  mu(0) = mu(1) * 50;
  PWConstCoefficient mu_func(mu);

  ParBilinearForm *a = new ParBilinearForm(fespace);
  a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

  //     Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  if (myid == 0) {
    cout << "matrix ... " << flush;
  }
  a->Assemble();

  HypreParMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
  if (myid == 0) {
    cout << "done." << endl;
    cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
  }

  HypreBoomerAMG *amg = new HypreBoomerAMG(A);

  if (amg_elast) {
    amg->SetElasticityOptions(fespace, false);
  } else {
    // amg->SetElasticityOptions(fespace);
    //  using this set system options seems to be ok though, still not a great
    //  PC though...
    amg->SetSystemsOptions(dim, Ordering::byVDIM == ordering);
  }

  IterativeSolver *solver;
  if (strcmp(solver_method, "CG") == 0) {
    solver = new CGSolver(MPI_COMM_WORLD);
  } else if (strcmp(solver_method, "SLI") == 0) {
    solver = new SLISolver(MPI_COMM_WORLD);
  } else {
    cout << "Bad solver choice: " << solver_method << endl;
    cout << "Must be either CG or SLI" << endl;
    return -1;
  }
  solver->SetRelTol(1e-12);
  solver->SetMaxIter(10000);
  mfem::IterativeSolver::PrintLevel pl;
  solver->SetPrintLevel(pl.Summary());
  solver->SetPreconditioner(*amg);
  solver->SetOperator(A);

  StopWatch timer;
  if (Mpi::Root()) {
    cout << "Size of linear system: " << A.Height() << endl;
    timer.Start();
  }
  solver->Mult(B, X);
  a->RecoverFEMSolution(X, *b, x);
  MPI_Barrier(MPI_COMM_WORLD);
  if (Mpi::Root()) {
    cout << "Solved in: " << timer.RealTime() << endl;
  }
  delete amg;
  delete a;
  delete b;
  if (fec) {
    delete fespace;
    delete fec;
  }
  delete pmesh;
  delete solver;

  return 0;
}
