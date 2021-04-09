import pandas as pd

pd.set_option('display.max_colwidth', -1)
names =  pd.read_csv('../datasets/sec__edgar_company_info.csv')
print('The shape: %d x %d' % names.shape)
names.head()

import re

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

from sklearn.feature_extraction.text import TfidfVectorizer

company_names = names['Company Name']
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(company_names)

print("density: %s" % ((tf_idf_matrix.nnz / (tf_idf_matrix.shape[0] * tf_idf_matrix.shape[1]) ) * 100))
print("shape: %s" % str(tf_idf_matrix.shape))

import numpy as np

print("max degree: %s" % (np.max(np.diff(tf_idf_matrix.indptr))))

from cuml.neighbors import NearestNeighbors

for m in ["cosine", "euclidean", "jaccard", "hellinger", "russelrao"
          "manhattan", "canberra", "chebyshev", "hamming", "jensenshannon",
          "kl_divergence"
        ]:
  nn = NearestNeighbors(n_neighbors=10, metric=m)
  nn.fit(tf_idf_matrix)

  import time

  s = time.time()
  nn.kneighbors(tf_idf_matrix)
  print("%s time: %s" % (m, (time.time() - s)))
