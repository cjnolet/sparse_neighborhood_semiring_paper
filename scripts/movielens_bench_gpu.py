if __name__ == "__main__":

  import cuml
  import cudf
  import cupy
  from cuml.neighbors import NearestNeighbors

  a = cudf.read_csv("../datasets/ratings.csv")
  b = cupy.sparse.coo_matrix((cupy.asarray(a["rating"].to_gpu_array()),(cupy.asarray(a["userId"].to_gpu_array()), cupy.asarray(a["movieId"].to_gpu_array()))))
  b = b.tocsr()[:,:]

  import time
  dists = ["cosine", "euclidean", "manhattan", "canberra", "chebyshev",
           "jaccard", "dice", "hamming", "jensenshannon", 
           "kl_divergence", "hellinger", "russelrao", "minkowski", "correlation"]


  for m in dists:
    nn = NearestNeighbors(n_neighbors=10, metric=m)#, verbose=logger.level_info, algo_params={'batch_size_index':40000, 'batch_size_query':40000})
    nn.fit(b)

    s = time.time()
    dist, inds = nn.kneighbors(b)
    print("%s time: %s" % (m, (time.time() - s)))

