import cupy
import cudf

df = cudf.read_csv("/datasets/cnolet/semiring_paper/datasets/docword.nytimes.txt", sep=" ", header=None, skiprows=3)
row = cupy.asarray(df["0"].to_gpu_array())
col = cupy.asarray(df["1"].to_gpu_array())
data = cupy.asarray(df["2"].to_gpu_array())
del df

a = cupy.sparse.coo_matrix((data, (row, col)), dtype='float32')

del row
del col
del data

b = cupy.sparse.csr_matrix(a)
del a

from cuml.feature_extraction.text import TfidfTransformer
c = TfidfTransformer().fit_transform(b)

d = cupy.diff(c.indptr)

print(str(c.shape))
print("density: %s" % (c.nnz / (c.shape[0] * c.shape[1])))
print("min %s" % cupy.max(d))
print("max %s" % cupy.min(d))

from cuml.neighbors import NearestNeighbors
from cuml.common import logger
logger.set_level(logger.level_debug)

del b

for m in [#"euclidean", 
          "manhattan", 
          "canberra", 
          #"jaccard", 
          "chebyshev", "hamming", "jensenshannon", "kl_divergence", 
          #"hellinger", "russelrao"
          #"cosine"
         ]:
    nn = NearestNeighbors(n_neighbors=5, metric=m)#, verbose=logger.level_debug, algo_params={'batch_size_index':40000, 'batch_size_query':40000})
    nn.fit(c)


    nnzs = cupy.diff(c.indptr)

    import time
    s = time.time()
    nn.kneighbors(c)
    print("%s Time: %ss" % (m, (time.time() - s)))
