if __name__ == "__main__":

  import sklearn 
  import pandas as pd
  import scipy 
  import scipy.sparse
  from sklearn.neighbors import NearestNeighbors

  a = pd.read_csv("../datasets/ratings.csv")

  b = scipy.sparse.coo_matrix((scipy.asarray(a["rating"]),(scipy.asarray(a["userId"]), scipy.asarray(a["movieId"]))))
  b = b.tocsr()[:,:]

  dists = ["cosine", "euclidean", "manhattan"]:

  for m in dists:
    import time
    nn = NearestNeighbors(n_neighbors=10, metric=m, n_jobs=-1)
    nn.fit(b)

    s = time.time()
    dist, inds = nn.kneighbors(b)

    print("%s time: %s" % (m, (time.time() - s)))

