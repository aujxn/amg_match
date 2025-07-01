import scipy

poly = 1
fem = 'cg'

for ref in range(2, 5):
    mat = scipy.sparse.load_npz(f"../data/lanl/{fem}/tokamakA_CG_Schur_ref{ref}.npz")
    n_cmps = scipy.sparse.csgraph.connected_components(mat)
    print(f'loaded refinement {ref} mat with {mat.shape} shape, {mat.nnz} nnz, and {n_cmps} connected comps.')
    bdy = []
    for i in range(mat.shape[0]):
        if mat.indptr[i] + 1 ==  mat.indptr[i+1]:
            bdy.append(i)
    print(f'found {len(bdy)} dirichlet nodes')
    bdy_file = f'{len(bdy)}\n'
    for i in bdy:
        bdy_file += f'{i}\n'

    with open(f"../data/lanl/{fem}/ref{ref}_p{poly}.bdy", "w") as f:
        f.write(bdy_file)

    scipy.io.mmwrite(f"../data/lanl/{fem}/ref{ref}_p{poly}.mtx", mat)

    with open(f"../data/lanl/{fem}/ref{ref}_p{poly}.rhs", "w") as f:
        for i in range(mat.shape[0]):
            f.write("0.0\n")
