***************
scikit-survival
***************

.. image:: https://img.shields.io/badge/license-GPLv3-blue.svg
  :target: COPYING
  :alt: License

.. image:: https://github.com/sebp/scikit-survival/workflows/Linux/badge.svg?branch=master
  :target: https://github.com/sebp/scikit-survival/actions?query=workflow%3ALinux+branch%3Amaster+
  :alt: Linux Build Status

.. image:: https://ci.appveyor.com/api/projects/status/github/sebp/scikit-survival?branch=master&svg=true
   :target: https://ci.appveyor.com/project/sebp/scikit-survival
   :alt: Windows Build Status on AppVeyor

.. image:: https://codecov.io/gh/sebp/scikit-survival/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/sebp/scikit-survival
  :alt: codecov

.. image:: https://api.codacy.com/project/badge/Grade/17242004cdf6422c9a1052bf1ec63104
   :target: https://www.codacy.com/app/sebp/scikit-survival?utm_source=github.com&utm_medium=referral&utm_content=sebp/scikit-survival&utm_campaign=badger
   :alt: Codacy Badge

.. image:: https://readthedocs.org/projects/scikit-survival/badge/?version=latest
  :target: https://scikit-survival.readthedocs.io/en/latest/
  :alt: readthedocs.org

.. image:: https://zenodo.org/badge/77409504.svg
   :target: https://zenodo.org/badge/latestdoi/77409504
   :alt: Digital Object Identifier (DOI)

scikit-survival is a Python module for `survival analysis`_
built on top of `scikit-learn <http://scikit-learn.org/>`_. It allows doing survival analysis
while utilizing the power of scikit-learn, e.g., for pre-processing or doing cross-validation.

=======================
About Survival Analysis
=======================

The objective in `survival analysis`_ (also referred to as reliability analysis in engineering)
is to establish a connection between covariates and the time of an event.
What makes survival analysis differ from traditional machine learning is the fact that
parts of the training data can only be partially observed – they are *censored*.

For instance, in a clinical study, patients are often monitored for a particular time period,
and events occurring in this particular period are recorded.
If a patient experiences an event, the exact time of the event can
be recorded – the patient’s record is uncensored. In contrast, right censored records
refer to patients that remained event-free during the study period and
it is unknown whether an event has or has not occurred after the study ended.
Consequently, survival analysis demands for models that take
this unique characteristic of such a dataset into account.

============
Requirements
============

- Python 3.5 or later
- cvxpy
- cvxopt
- joblib
- numexpr
- numpy 1.12 or later
- osqp
- pandas 0.21 or later
- scikit-learn 0.22
- scipy 1.0 or later
- C/C++ compiler

============
Installation
============

The easiest way to install scikit-survival is to use
`Anaconda <https://www.anaconda.com/distribution/>`_ by running::

  conda install -c sebp scikit-survival

Alternatively, you can install scikit-survival from source
following `this guide <https://scikit-survival.readthedocs.io/en/latest/install.html#from-source>`_.

========
Examples
========

The following examples are available as `Jupyter notebook <https://jupyter.org/>`_:

* `Introduction to Survival Analysis with scikit-survival <https://nbviewer.jupyter.org/github/sebp/scikit-survival/blob/master/examples/00-introduction.ipynb>`_
* `Pitfalls when Evaluating Survival Models <https://nbviewer.jupyter.org/github/sebp/scikit-survival/blob/master/examples/evaluating-survival-models.ipynb>`_
* `Introduction to Kernel Survival Support Vector Machines <https://nbviewer.jupyter.org/github/sebp/scikit-survival/blob/master/examples/survival-svm.ipynb>`_
* `Using Random Survival Forests <https://nbviewer.jupyter.org/github/sebp/scikit-survival/blob/master/examples/random-survival-forest.ipynb>`_

================
Help and Support
================

**Documentation**

- HTML documentation for the latest release: https://scikit-survival.readthedocs.io/en/stable/
- HTML documentation for the development version (master branch): https://scikit-survival.readthedocs.io/en/latest/
- For a list of notable changes, see the `release notes <https://scikit-survival.readthedocs.io/en/stable/release_notes.html>`_.

**Bug reports**

- If you encountered a problem, please submit a
  `bug report <https://github.com/sebp/scikit-survival/issues/new?template=bug_report.md>`_.

**Questions**

- For general theoretical or methodological questions on survival analysis, please use
  `Cross Validated <https://stats.stackexchange.com/questions/tagged/survival>`_.

============
Contributing
============

New contributors are always welcome. Please have a look at the
`contributing guidelines <https://scikit-survival.readthedocs.io/en/latest/contributing.html>`_
on how to get started and to make sure your code complies with our guidelines.

==========
References
==========

Please cite the following papers if you are using **scikit-survival**.

1. Pölsterl, S., Navab, N., and Katouzian, A.,
`Fast Training of Support Vector Machines for Survival Analysis <http://link.springer.com/chapter/10.1007/978-3-319-23525-7_15>`_.
Machine Learning and Knowledge Discovery in Databases: European Conference,
ECML PKDD 2015, Porto, Portugal,
Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)

2. Pölsterl, S., Navab, N., and Katouzian, A.,
`An Efficient Training Algorithm for Kernel Survival Support Vector Machines <https://arxiv.org/abs/1611.07054>`_.
4th Workshop on Machine Learning in Life Sciences,
23 September 2016, Riva del Garda, Italy

3. Pölsterl, S., Gupta, P., Wang, L., Conjeti, S., Katouzian, A., and Navab, N.,
`Heterogeneous ensembles for predicting survival of metastatic, castrate-resistant prostate cancer patients <http://doi.org/10.12688/f1000research.8231.1>`_.
F1000Research, vol. 5, no. 2676 (2016).

.. _survival analysis: https://en.wikipedia.org/wiki/Survival_analysis
