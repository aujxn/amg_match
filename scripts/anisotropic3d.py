from ngsolve import *
from netgen.occ import *
from scipy import io
import scipy.sparse as sp
import random

box = Box((-1, -1, -1), (1, 1, 1))
box.faces.name = "out"
mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=0.1))
for i in range(5):
    mesh.Refine()

fes = H1(mesh, order=1, dirichlet="out")
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
new_rows = []
new_cols = []
new_vals = []

for row, col, val in zip(rows, cols, vals):
    if fes.FreeDofs()[row] and fes.FreeDofs()[col]:
        new_rows.append(row)
        new_cols.append(col)
        new_vals.append(val)
    elif row == col:
        new_rows.append(row)
        new_cols.append(col)
        new_vals.append(1.0)

A = sp.csr_matrix((new_vals,(new_rows,new_cols)))
io.mmwrite(f"anisotropic3d{A.shape[0]}.mtx", A); 
