# Reproducibility Artifacts

This repository contains artifacts and reproducibility information for the paper "Semiring Primitives for Sparse Neighborhood Methods on the GPU". The benchmarks presented in the paper were performed on a DGX1 w/ V100 GPUs using driver version 450.102.04 and CUDA toolkit version 11.0. More environment info is provided [here](https://github.com/cjnolet/sparse_neighborhood_semiring_paper/blob/main/environment_info.txt). We did not execute any benchmarks concurrently and made sure we had exclusive access to the system (e.g. aside from regular system maintenance and administration, no other scripts were executing on the system that would heavily tax the CPU or GPU resources).

The following steps should produce benchmarks in the paper.

1. Clone this repository

2. Build and install RAPIDS cuML from source using [these](https://github.com/rapidsai/cuml/blob/branch-0.19/BUILD.md#installing-from-source) instructions. Most of the dependencies can be installed through Anaconda by using the conda environment configuration yaml file outlined [here](https://github.com/rapidsai/cuml/blob/branch-0.19/BUILD.md#setting-up-your-build-environment). Our benchmark code is based on the 0.19 version of any RAPIDS packages.

You will need to build cuML by cloning the following two branches:
```bash
git clone --single-branch --branch semiring_primitives_baseline git@github.com:cjnolet/cuml.git cuml_baseline
```
```bash
git clone --single-branch --branch semiring_primitives_optim git@github.com:cjnolet/cuml.git cuml_optim
```

3. Download the datsets and place/decompress in a directory `datasets` in the root of this repository:
  - [The MovieLense 20M Ratings Dataset](https://files.grouplens.org/datasets/movielens/ml-20m.zip)
  - [NY Times Bag of Words Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz)
  - [SEC EDGAR Company Names Dataset](https://www.kaggle.com/dattapiy/sec-edgar-companies-list)
  - [scRNA 70k Lung Cell Dataset](https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x.sparse.h5ad)

4. The scripts provided in the `scripts` directory of this repository will execute the benchmarks using the `NearestNeighbors` estimator. 
  - The scripts that run the GPU implementations have the `_gpu` suffix and execute in cuML. 
  - All other scripts use Scikit-learn on the CPU.
