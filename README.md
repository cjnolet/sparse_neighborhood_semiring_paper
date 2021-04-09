# Reproducibility Artifacts

This repository contains artifacts and reproducibility information for the paper "Semiring Primitives for Sparse Neighborhood Methods on the GPU"

The following steps should produce benchmarks of our optimized GPU semirings implementation:

1. Clone this repository
2.Build and install RAPIDS cuML, which can be done through Anaconda. The instructions to build and install cuML are located [here](https://github.com/rapidsai/cuml/blob/branch-0.19/BUILD.md).

You will need to build cuML by cloning the following two branches:
```bash
git clone --single-branch --branch semiring_primitives_baseline git@github.com:cjnolet/cuml.git cuml_baseline
```
```bash
git clone --single-branch --branch semiring_primitives_optim git@github.com:cjnolet/cuml.git cuml_optim
```

3. Download the datsets and place in a directory `datasets` in the root of this repository
  - The MovieLense 20M Dataset: https://grouplens.org/datasets/movielens/
  - NY Times BoW Dataset: https://archive.ics.uci.edu/ml/datasets/bag+of+words
  - SEC EDGAR Company names Dataset: https://www.kaggle.com/dattapiy/sec-edgar-companies-list
  - scRNA Dataset: https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x.sparse.h5ad

4. The scripts provided in the `scripts` directory of this repository should execute very simple and straightforward benchmarks using the `NearestNeighbors` estimator in Scikit-learn. The scripts that run the GPU implementations have the `_gpu` suffix. 
