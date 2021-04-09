import scanpy as sc
import anndata

from sklearn.neighbors import NearestNeighbors 

adata = sc.read("../datasets/krasnow_hlca_10x.sparse.h5ad")

import time


for m in ["cosine", "euclidean", "manhattan"]:
  nn = NearestNeighbors(n_neighbors=10, metric=m, n_jobs=-1)
  nn.fit(adata.X)

  s = time.time()
  nn.kneighbors(adata.X)
  print("%s sklearn finished in: %ds" % (m, (time.time() - s)))
