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
c = TfidfTransformer().fit_transform(b).get()

if __name__ == "__main__":
  from sklearn.neighbors import NearestNeighbors

  for m in ["cosine", "euclidean", "manhattan"]:
    
    nn = NearestNeighbors(n_neighbors=5, metric=m, n_jobs=-1)
    nn.fit(c)

    import time
    s = time.time()
    nn.kneighbors(c)
    print("%s Time: %ss" % (m, (time.time() - s)))
