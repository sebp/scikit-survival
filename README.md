# Fast Training of Support Vector Machines for Survival Analysis

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
[![Build Status](https://travis-ci.org/tum-camp/survival-support-vector-machine.svg)](https://travis-ci.org/tum-camp/survival-support-vector-machine)
[![codecov](https://codecov.io/gh/tum-camp/survival-support-vector-machine/branch/master/graph/badge.svg)](https://codecov.io/gh/tum-camp/survival-support-vector-machine)
[![DOI](https://zenodo.org/badge/16868/tum-camp/survival-support-vector-machine.svg)](https://zenodo.org/badge/latestdoi/16868/tum-camp/survival-support-vector-machine)

This repository contains an efficient implementation of *Survival Support
Vector Machines* as proposed in

> Pölsterl, S., Navab, N., and Katouzian, A.,
> *[Fast Training of Support Vector Machines for Survival Analysis](http://link.springer.com/chapter/10.1007/978-3-319-23525-7_15)*,
> Machine Learning and Knowledge Discovery in Databases: European Conference,
> ECML PKDD 2015, Porto, Portugal,
> Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)

> Pölsterl, S., Navab, N., and Katouzian, A.,
> *An Efficient Training Algorithm for Kernel Survival Support Vector Machines*
> 4th Workshop on Machine Learning in Life Sciences,
> 23 September 2016, Riva del Garda, Italy


## Requirements

- Python 3.4 or later
- cvxpy
- cvxopt
- numexpr
- numpy 1.9 or later
- pandas 0.17.0 or later
- scikit-learn 0.17
- scipy 0.16 or later
- C/C++ compiler
- ipython (optional)
- seaborn (optional)

## Getting Started

The easiest way to get started is to install [Anaconda](https://store.continuum.io/cshop/anaconda/)
and setup an environment. To create a new environment, named `ssvm`, run:

```
conda create -n ssvm python=3 --file requirements.txt
```

To work in this environment, ``activate`` it as follows:

```
source activate ssvm
```

If you are on Windows, run the above command without the ``source`` in the beginning.

Once you setup your build environment, you have to compile the C/C++
extensions and install the package by running:

```
python setup.py install
```

Alternatively, if you want to use the package without installing it,
you can compile the extensions in place by running:

```
python setup.py build_ext --inplace
```

To check everything is setup correctly run the test suite by executing:

```
nosetests
```

## Examples

A [simple example][Notebook] on how to use our implementation of Survival Support
Vector Machines is described in an [IPython/Jupyter notebook](https://jupyter.org/).

A more elaborate script that can be used to reproduce the results in the paper
is `grid_search_parallel.py` in the examples directory.
When running it you need to specify the algorithm (`--method`)
and dataset (`--dataset`) to use:

```bash
# Start IPython cluster to run grid search in parallel
ipcluster start &
# Run cross-validation. Results are stored in results-veteran-l2_ranking.csv
python examples/grid_search_parallel.py --dataset veteran --method l2_ranking
# Find best hyper-parameter configuration and visualize the results
python examples/plot-performance.py -o results.pdf results-veteran-l2_ranking.csv
```
The example above requires the [Ipython](http://ipython.org) and
[seaborn](http://stanford.edu/~mwaskom/software/seaborn/) packages.

The script runs cross-validation with 200 randomly selected 50/50 splits of
the dataset. This is repeated for each possible configuration of hyper-parameters
(see *Methods* section below). Each time the following performance measures
are computed:
  1. Harrell's concordance index, and
  2. root mean squared error (RMSE) with respect to uncensored records.

The output is a CSV file that contains the performance on the test set for
each fold and hyper-parameter configuration. Additional options of the script
are available when running the script with the ``--help`` argument.

### Methods

The grid search for all methods contains 13 configurations for the
regularization parameter alpha: 2<sup>i</sup>, from i = -12 to 12 in steps of 2.
When using the hybrid ranking-regression loss, an additional 21 configurations
for the ratio between the two losses are considered:
0.05 to 0.95 in steps of 0.05.

| Method | Description | rank_ratio |
| ------ | ----------- | ---------- |
| l1 | Naive implementation of Survival SVM using hinge loss. | - |
| l2_ranking | Fast implementation of Survival SVM using squared hinge loss (ranking objective only). | 1.0 |
| l2_regression | Fast implementation of Survival SVM using squared loss (regression objective only). | 0.0 |
| l2_ranking_regression | Fast implementation of Survival SVM using hybrid of squared hinge loss for ranking and squared loss for regression. | 0.05, 0.10, …, 0.95 |


### Datesets

The repository contains four datasets that are freely available and can
be used to reproduce the results in the paper.

| Dataset | Description | Samples | Features | Events | Outcome |
| ------- | ----------- | ------- | -------- | ------ | ------- |
| actg320_aids or actg320_death | [AIDS study][Hosmer2008] | 1,151 | 13 | 96 (8.3%) | AIDS defining event or death |
| breast-cancer | [Breast cancer][Desmedt2007] | 198 | 80 | 62 (31.3%) | Distant metastases |
| veteran | [Veteran's Lung Cancer][Kalbfleisch2008] | 137 | 6 | 128 (93.4%) | Death |
| whas500 | [Worcester Heart Attack Study][Hosmer2008] | 500 | 14 | 215 (43.0%) | Death|


## Documentation

The source code is thoroughly documented and a HTML version of the API documentation
is available at https://tum-camp.github.io/survival-support-vector-machine/

You can generate the documentation yourself using [Sphinx](http://sphinx-doc.org/) 1.4 or later.

```bash
cd doc
PYTHONPATH="..:sphinxext" sphinx-autogen api.rst
make html
xdg-open _build/html/index.html
```

[Desmedt2007]: http://dx.doi.org/10.1158/1078-0432.CCR-06-2765 "Desmedt, C., Piette, F., Loi et al.: Strong Time Dependence of the 76-Gene Prognostic Signature for Node-Negative Breast Cancer Patients in the TRANSBIG Multicenter Independent Validation Series. Clin. Cancer Res. 13(11), 3207–14 (2007)"

[Hosmer2008]: http://www.wiley.com/WileyCDA/WileyTitle/productCd-0471754994.html "Hosmer, D., Lemeshow, S., May, S.: Applied Survival Analysis: Regression Modeling of Time to Event Data. John Wiley & Sons, Inc. (2008)"

[Kalbfleisch2008]: http://www.wiley.com/WileyCDA/WileyTitle/productCd-047136357X.html "Kalbfleisch, J.D., Prentice, R.L.: The Statistical Analysis of Failure Time Data. John Wiley & Sons, Inc. (2002)"

[Notebook]: http://nbviewer.ipython.org/github/tum-camp/survival-support-vector-machine/blob/master/examples/survival-svm.ipynb "IPython notebook example"
