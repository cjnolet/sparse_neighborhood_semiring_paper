import cudf
from cuml.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
class StringMatcher:
    def __init__(self, ground_truth, col, ngram_range=(1, 2), topk=5, lower_bound=0.5):
        self.ground_truth = ground_truth
        self.vec = TfidfVectorizer(ngram_range=ngram_range)
        self.topk = topk
        self.lower_bound = lower_bound
        self.col = col
    def get_matches_df(self, distances, indices, names):
        match_scores = 1 - distances
        gt_indices = indices
        gt_names = self.ground_truth[self.col].to_array()
        res_df = []
        for i in range(distances.shape[0]):
            name = names[i]
            for j in range(self.topk):
                if match_scores[i, j] > self.lower_bound:
                    res_df.append({self.col: name, "Ground Truth": gt_names[gt_indices[i, j]],
                                   "match_score": match_scores[i, j], "match_rank": j + 1})
        res_df = cudf.DataFrame(res_df)
        res_df = cudf.DataFrame({self.col: names}).merge(res_df, on=self.col, how="left")
        return res_df
    def match(self, names_to_match, m = 'cosine'):
        print(f"Matching {names_to_match.shape[0]} names to {self.ground_truth.shape[0]} names...")
        self.vec.fit(names_to_match.append(self.ground_truth)[self.col])
        X_gt = self.vec.transform(self.ground_truth[self.col]).get()
        X = self.vec.transform(names_to_match[self.col]).get()

        import time


        nn = NearestNeighbors(n_neighbors=self.topk, metric=m, n_jobs=-1)#, algo_params={"batch_size_index": 40000, "batch_size_query":40000})
        nn.fit(X_gt)
        
        s = time.time()
        distances, indices = nn.kneighbors(X)
        print("%s Kneighors time: %s" % (m, (time.time() - s)))

        return self.get_matches_df(distances, indices, names_to_match[self.col].to_array())

import time

gt_df = cudf.read_csv("../datasets/sec__edgar_company_info.csv")

sm = StringMatcher(gt_df, col="Company Name", ngram_range=(1, 2), topk=5, lower_bound=0.5)

names_to_match = gt_df#[gt_df["Line Number"] % 6 == 0].reset_index(drop=True)

s = time.time()

for m in ["euclidean", "manhattan"]:
  
  print("Running metric %s" % m)
  res_df = sm.match(names_to_match, m = m)

  print("%s Time: %s" % (m, (time.time() - s)))

#ground_truth = gt_df[gt_df["Line Number"] % 6 != 0].reset_index(drop=True)

#print(res_df.head())

