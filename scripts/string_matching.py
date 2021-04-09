
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn


class StringMatcher:
    def __init__(self, ground_truth, col, ngram_range=(1, 2), topk=5, lower_bound=0.5):
        self.ground_truth = ground_truth
        self.vec = TfidfVectorizer(ngram_range=ngram_range)
        self.topk = topk
        self.lower_bound = lower_bound
        self.col = col
        
    def get_matches_df(self, sparse_matrix, names):
        non_zeros = sparse_matrix.nonzero()

        name_indices = non_zeros[0]
        gt_indices = non_zeros[1]

        left_side = np.empty(gt_indices.size, dtype=object)
        right_side = np.empty(gt_indices.size, dtype=object)
        match_score = np.zeros(gt_indices.size)

        for index in range(gt_indices.size):
            left_side[index] = names.values[name_indices[index]]
            right_side[index] = self.ground_truth[self.col].values[gt_indices[index]]
            match_score[index] = sparse_matrix.data[index]

        res_df = pd.DataFrame({self.col: left_side,
                               'Ground Truth': right_side,
                               'match_score': match_score})

        res_df["match_rank"] = res_df.groupby(self.col)["match_score"].rank(ascending=False)

        res_df = pd.DataFrame(names).merge(res_df, on=self.col, how="left")
        return res_df


    def match(self, names_to_match, n_threads=16):
        print(f"Matching {names_to_match.shape[0]} names to {self.ground_truth.shape[0]} names...")
        
        self.vec.fit(names_to_match.append(self.ground_truth)[self.col])
        X_gt = self.vec.transform(self.ground_truth[self.col])
        X = self.vec.transform(names_to_match[self.col])
        
        import time
        s = time.time()
        d = awesome_cossim_topn(X, X_gt.T, self.topk, self.lower_bound, use_threads=True, n_jobs=n_threads)
        print("cossim topn match time %s" % (time.time() - s))
        
        return self.get_matches_df(d, names_to_match[self.col])

import time

gt_df = pd.read_csv("../datasets/sec__edgar_company_info.csv")

sm = StringMatcher(gt_df, col="Company Name", ngram_range=(1, 2), topk=5, lower_bound=0.5)

names_to_match = gt_df #[gt_df["Line Number"] % 6 == 0].reset_index(drop=True)

s = time.time()

res_df = sm.match(names_to_match)

print("Time: %s" % (time.time() - s))

ground_truth = gt_df[gt_df["Line Number"] % 6 != 0].reset_index(drop=True)

print(res_df.head())

