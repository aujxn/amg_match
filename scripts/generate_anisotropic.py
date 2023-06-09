import mfem.ser as mfem
import numpy as np

class DiffusionCoefficient(mfem.MatrixPyCoefficient):

    def __init__(self):
        mfem.MatrixPyCoefficient.__init__(self, 2)

        theta = 55.0 / 180.0 * np.pi
        ct = np.cos(theta)
        st = np.sin(theta)
        b = np.array([[ct],[st]])
        epsilon = 0.001

        self.coef = epsilon * np.eye(2) + b @ b.T 

    def EvalValue(self, x):
        return self.coef

def run(order=1,  meshfile=''):
    '''
    run ex0
    '''

    mesh = mfem.Mesh(meshfile, 1, 1)
    mesh.UniformRefinement()

    fec = mfem.H1_FECollection(order,  mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec)
    print('Number of finite element unknowns: ' +
          str(fespace.GetTrueVSize()))

    boundary_dofs = mfem.intArray()
    fespace.GetBoundaryTrueDofs(boundary_dofs)

    # 5. Define the solution x as a finite element grid function in fespace. Set
    #    the initial guess to zero, which also sets the boundary conditions.
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # 6. Set up the linear form b(.) corresponding to the right-hand side.
    one = mfem.ConstantCoefficient(1.0)
    b = mfem.LinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()

    # 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a.Assemble()

    # 8. Form the linear system A X = B. This includes eliminating boundary
    #    conditions, applying AMR constraints, and other transformations.
    A = mfem.SparseMatrix()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B)
    print("Size of linear system: " + str(A.Height()))

    # 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
    M = mfem.GSSmoother(A)
    mfem.PCG(A, M, B, X, 1, 200, 1e-12, 0.0)

    # 10. Recover the solution x as a grid function and save to file. The output
    #     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
    a.RecoverFEMSolution(X, b, x)
    x.Save('sol.gf')
    mesh.Save('mesh.mesh')


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Ex1 (Laplace Problem)')
    parser.add_argument('-m', '--mesh',
                        default='star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree) or -1 for isoparametric space.")

    args = parser.parse_args()
    parser.print_options(args)

    order = args.order
    meshfile = expanduser(
        join(os.path.dirname(__file__), '..', 'data', args.mesh))

    run(order=order,
        meshfile=meshfile)
