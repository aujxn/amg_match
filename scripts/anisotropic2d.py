from ngsolve import *
from netgen.geom2d import unit_square
from scipy import io
import scipy.sparse as sp
import random

for i in range(4):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    for j in range(i + 2):
        mesh.Refine()

    fes = H1(mesh, order=1, dirichlet=".*")
    u,v = fes.TnT()

    theta = Parameter(random.random())

    epsilon = CoefficientFunction(
            (0.001, 0,
             0, 0.001),
            dims=(2,2))
    b = CoefficientFunction(
            (cos(theta),
             sin(theta)),
            dims=(2,1))
    coef = epsilon + b * b.trans

    with TaskManager():
        a = BilinearForm(coef * grad(u) * grad(v) * dx)
        a.Assemble()

    rows,cols,vals = a.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols)))
    io.mmwrite(f"anisotropic2d{A.shape[0]}.mtx", A); 
