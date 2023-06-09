import scipy as sp
import scipy.io
import networkx as nx

G = nx.from_scipy_sparse_array(sp.io.mmread("test_matrices/m_t1/m_t1.mtx"))

linefeed = chr(10)
s = linefeed.join(nx.generate_gexf(G))

output = open("m_t1.gexf", "a")

for line in nx.generate_gexf(G):
    print(line)
    output.write(line)
