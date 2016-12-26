# scikit-survival

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
[![Build Status](https://travis-ci.org/tum-camp/survival-support-vector-machine.svg)](https://travis-ci.org/tum-camp/survival-support-vector-machine)
[![codecov](https://codecov.io/gh/tum-camp/survival-support-vector-machine/branch/master/graph/badge.svg)](https://codecov.io/gh/tum-camp/survival-support-vector-machine)
[![DOI](https://zenodo.org/badge/37088896.svg)](https://zenodo.org/badge/latestdoi/37088896)

scikit-survival is a Python module for [survival analysis][survival_analysis]
built on top of [scikit-learn](http://scikit-learn.org/). It allows doing survival analysis
while utilizing the power of scikit-learn, e.g., for pre-processing or doing cross-validation.

## About Survival Analysis

The objective in [survival analysis][survival_analysis] (also referred to as reliability analysis in engineering)
is to establish a connection between covariates and the time of an event.
What makes survival analysis differ from traditional machine learning is the fact that
parts of the training data can only be partially observed – they are *censored*.

For instance, in a clinical study, patients are often monitored for a particular time period,
and events occurring in this particular period are recorded.
If a patient experiences an event, the exact time of the event can
be recorded – the patient’s record is uncensored. In contrast, right censored records
refer to patients that remained event-free during the study period and
it is unknown whether an event has or has not occurred after the study ended.
Consequently, survival models demand for models that take
this unique characteristic of such a dataset into account.


## Requirements

- Python 3.4 or later
- cvxpy
- cvxopt
- numexpr
- numpy 1.10 or later
- pandas 0.18
- scikit-learn 0.18
- scipy 0.17 or later
- C/C++ compiler

## Installation

The easiest way to get started is to install [Anaconda](https://store.continuum.io/cshop/anaconda/)
and setup an environment.

```
conda install -c sebp survival-svm
```

### Installing from source

First, create a new environment, named `ssvm`:

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

A [simple example][Notebook] on how to use Survival Support
Vector Machines is described in an [Jupyter notebook](https://jupyter.org/).


## Documentation

The source code is thoroughly documented and a HTML version of the API documentation
is available at https://tum-camp.github.io/survival-support-vector-machine/

You can generate the documentation yourself using [Sphinx](http://sphinx-doc.org/) 1.4 or later.

```bash
cd doc
PYTHONPATH=".." sphinx-autogen api.rst
make html
xdg-open _build/html/index.html
```

[Notebook]: http://nbviewer.ipython.org/github/tum-camp/survival-support-vector-machine/blob/master/examples/survival-svm.ipynb "IPython notebook example"

### References

Please cite the following papers if you are using **scikit-survival**.

> Pölsterl, S., Navab, N., and Katouzian, A.,
> *[Fast Training of Support Vector Machines for Survival Analysis](http://link.springer.com/chapter/10.1007/978-3-319-23525-7_15)*.
> Machine Learning and Knowledge Discovery in Databases: European Conference,
> ECML PKDD 2015, Porto, Portugal,
> Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)

> Pölsterl, S., Navab, N., and Katouzian, A.,
> *[An Efficient Training Algorithm for Kernel Survival Support Vector Machines](https://arxiv.org/abs/1611.07054)*.
> 4th Workshop on Machine Learning in Life Sciences,
> 23 September 2016, Riva del Garda, Italy

> Pölsterl, S., Gupta, P., Wang, L., Conjeti, S., Katouzian, A., and Navab, N.,
> *[Heterogeneous ensembles for predicting survival of metastatic, castrate-resistant prostate cancer patients](http://doi.org/10.12688/f1000research.8231.1)*.
> F1000Research, vol. 5, no. 2676 (2016).

[survival_analysis]: https://en.wikipedia.org/wiki/Survival_analysis