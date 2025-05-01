import numpy as np
import time
import glob
import pyamg
from scipy.io import mmread
from pyamg.aggregation import adaptive_sa_solver, smoothed_aggregation_solver

def read_gf(file, skip=0):
    with open(file) as f:
        vals = []
        for i, line in enumerate(f.readlines()):
            if i < skip:
                continue
            for val in line.split(" "):
                vals.append(float(val))

    return np.array(vals)

for r in range(2,6):
    mat = mmread(f"../data/elasticity/{r}/elasticity_3d.mtx")
    mat = mat.tobsr(blocksize=(3,3))
    #mat = mat.tocsr()
    b = read_gf(f"../data/elasticity/{r}/elasticity_3d.rhs")

    rbm_files = glob.glob(f"../data/elasticity/{r}/rbm_*.gf")
    rbms = np.zeros((mat.shape[0], 6))
    for col, file in enumerate(rbm_files):
        vals = read_gf(file, 5)
        if len(vals) != mat.shape[0]:
            print(f"rbm isn't correct size: {len(vals)} should be {mat.shape[0]}")
            exit(-1)
        rbms[:,col] = vals[:]

    [asa,work] = adaptive_sa_solver(mat, num_candidates=6, initial_candidates=rbms, candidate_iters=5, improvement_iters=0)
    #asa = smoothed_aggregation_solver(mat, rbms)

    print(asa)

    x0 = np.random.rand(mat.shape[0])
    r0 = np.linalg.norm(b - mat @ x0)
    i = 0

    def callback(x_i):
        global i 
        i += 1
        if i % 50 == 0:
            r = b - mat @ x_i
            print(f"{i}: {np.linalg.norm(r) / r0:.2e}")


    res = []
    tol = 1e-6
    start_time = time.perf_counter()
    x_sli, info_sli = asa.solve(b=b, x0=x0, residuals=res, return_info=True, maxiter=5000, tol=tol, callback=callback)
    conv_rate = (res[-1] / res[0]) ** (1. / len(res))
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"SLI solved in {len(res)} iters ({execution_time:.4f} seconds) with final relative residual {res[-1] / res[0] :.2e} and avg reduction factor {conv_rate:.2}")

    i = 0
    res = []
    start_time = time.perf_counter()
    x_cg, info_cg = asa.solve(b=b, x0=x0, residuals=res, return_info=True, maxiter=5000, tol=tol, accel="cg", callback=callback)
    conv_rate = (res[-1] / res[0]) ** (1. / len(res))
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"CG solved in {len(res)} iters ({execution_time:.4f} seconds) with final relative residual {res[-1] / res[0] :.2e} and avg reduction factor {conv_rate:.2}")
