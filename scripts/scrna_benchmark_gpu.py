import scanpy as sc
import anndata

import cupy as cp
from cuml.neighbors import NearestNeighbors 

adata = sc.read("../datasets/krasnow_hlca_10x.sparse.h5ad")

import numpy as np

b = np.diff(adata.X.indptr)

print("min %s" % np.min(b))
print("max %s" % np.max(b))

import time

for m in [#"euclidean", 
          #"manhattan", 
          #"canberra", 
          #"jaccard", 
         # "chebyshev", 
         # "hamming", 
         # "jensenshannon", 
          "kl_divergence", 
          #"hellinger", 
          #"russelrao", 
          #"dice", 
         # "minkowski"
          ]:
  nn = NearestNeighbors(n_neighbors=10, metric=m)#, algo_params={'batch_size_index':40000, 'batch_size_query':40000})
  nn.fit(adata.X)

  s = time.time()
  nn.kneighbors(adata.X)
  print("%s finished in: %ds" % (m, (time.time() - s)))
