import scipy.io
a = scipy.io.loadmat("../datasets/mc2depi.mat")
d = a["Problem"][0][0][2].tocsr()

from cuml.neighbors import NearestNeighbors
import numpy as np
c = np.diff(d.indptr)


import pandas as pd
hist, bin_edges = np.histogram(c, bins=5000, density=True)
dx = bin_edges[1] - bin_edges[0]
F1 = np.cumsum(hist) * dx
be = bin_edges[1:]
filtered_x = be[(F1>0.0) & (F1<=0.99)]
filtered_y = F1[(F1>0.0)& (F1<=0.99)]
cdf = {"x": filtered_x, "y": filtered_y}

pd.DataFrame(cdf).to_csv('epidemiology_degree_cdf.csv')

for m in ["cosine", "euclidean", "manhattan", "jaccard", "canberra", "chebyshev", "hamming", "hellinger", "jensenshannon", "kl_divergence"]:

  nn = NearestNeighbors(n_neighbors=10, metric='cosine')
  nn.fit(d)
  import time
  s = time.time()
  nn.kneighbors(d)
  print("%s Time: %s" % (m, (time.time() - s)))
