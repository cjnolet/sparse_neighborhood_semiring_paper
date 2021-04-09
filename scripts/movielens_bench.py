#! /share/software/miniconda3/envs/cuml_019_022220/bin/python

if __name__ == "__main__":

  import sklearn 
  import pandas as pd
  import scipy 
  import scipy.sparse
  from sklearn.neighbors import NearestNeighbors

  print("Reading CSV into cudf")
  a = pd.read_csv("../datasets/ratings.csv")

  print("Converting to scipy sparse matrix")
  b = scipy.sparse.coo_matrix((scipy.asarray(a["rating"]),(scipy.asarray(a["userId"]), scipy.asarray(a["movieId"]))))
  b = b.tocsr()[:500,:]

  print("b: %s" % b)

  print("Running nearest neighbors")

  dists = [#"cosine", "euclidean",
           "manhattan", #"canberra", #"jaccard",
           #"dice", 
           #"hellinger", 

           #"chebyshev"

          ]


  for m in dists:
    import time
    nn = NearestNeighbors(n_neighbors=10, metric=m, n_jobs=-1)

    print("Running fit()")
    nn.fit(b)

    s = time.time()
    print("Running kneighbors")
    dist, inds = nn.kneighbors(b)

    print(str(dist))

    print("%s time: %s" % (m, (time.time() - s)))

