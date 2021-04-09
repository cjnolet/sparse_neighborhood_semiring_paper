if __name__ == "__main__":

  import cuml
  import cudf
  import cupy
  from cuml.neighbors import NearestNeighbors

  print("Reading CSV into cudf")
  a = cudf.read_csv("../datasets/ratings.csv")

  print("Converting to cupy sparse matrix")
  b = cupy.sparse.coo_matrix((cupy.asarray(a["rating"].to_gpu_array()),(cupy.asarray(a["userId"].to_gpu_array()), cupy.asarray(a["movieId"].to_gpu_array()))))
  b = b.tocsr()[:,:]


 
  print("density: %s" % ((b.nnz / (b.shape[0] * b.shape[1])) * 100))
  print(str(b.shape))
  c = cupy.diff(b.indptr)
  print("b: %s" % len(c[c<300]))
  print(str(cupy.max(c)))
  print(str(cupy.min(c)))
  print("Running nearest neighbors")

  import time
  from cuml.common import logger
  logger.set_level(logger.level_info)

  dists = [#"cosine", "euclidean", 
           #"manhattan", "canberra", 
           
           #"chebyshev",
           #"jaccard", 
           #"dice", 
          # "hamming", "jensenshannon", 
          "kl_divergence", 
           #"hellinger",
           #"russelrao"
          # "minkowski"

          ]


  for m in dists:
    nn = NearestNeighbors(n_neighbors=10, metric=m)#, verbose=logger.level_info, algo_params={'batch_size_index':40000, 'batch_size_query':40000})
    s = time.time()

    print("Running fit()")
    nn.fit(b)

    print("Running kneighbors")
    dist, inds = nn.kneighbors(b)

    print(str(dist))

    print("%s time: %s" % (m, (time.time() - s)))

