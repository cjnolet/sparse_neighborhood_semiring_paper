import cupy
import cudf

df = cudf.read_csv("../datasets/docword.nytimes.txt", sep=" ", header=None, skiprows=3)
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

from cuml.neighbors import NearestNeighbors

del b

for m in ["cosine", "euclidean", "manhattan", "canberra", "jaccard", 
          "chebyshev", "hamming", "jensenshannon", "kl_divergence", 
          "hellinger", "russelrao", "dice", "minkowski", "correlation"]:
    nn = NearestNeighbors(n_neighbors=5, metric=m)
    nn.fit(c)

    import time
    s = time.time()
    nn.kneighbors(c)
    print("%s Time: %ss" % (m, (time.time() - s)))

