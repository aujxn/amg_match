from ngsolve import *
from netgen.occ import *
from scipy import io
import scipy.sparse as sp
import random

for i in range(3):
    box = Box((-1, -1, -1), (1, 1, 1))
    mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=0.1))
    for j in range(i):
        mesh.Refine()

    fes = H1(mesh, order=1, dirichlet=".*")
    u,v = fes.TnT()

    theta = Parameter(random.random())
    phi = Parameter(random.random())

    epsilon = CoefficientFunction(
            (0.001, 0, 0,
             0, 0.001, 0,
             0, 0, 0.001),
            dims=(3,3))
    b = CoefficientFunction(
            (cos(theta) * cos(phi),
             sin(theta) * cos(phi),
             sin(phi)),
            dims=(3,1))
    coef = epsilon + b * b.trans

    with TaskManager():
        a = BilinearForm(coef * grad(u) * grad(v) * dx)
        a.Assemble()

    rows,cols,vals = a.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols)))
    io.mmwrite(f"anisotropic3d{A.shape[0]}.mtx", A); 
