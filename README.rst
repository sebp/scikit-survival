|License| |Docs| |DOI|

|build-tests| |build-windows| |Codecov| |Codacy|

***************
scikit-survival
***************

scikit-survival is a Python module for `survival analysis`_
built on top of `scikit-learn <https://scikit-learn.org/>`_. It allows doing survival analysis
while utilizing the power of scikit-learn, e.g., for pre-processing or doing cross-validation.

=======================
About Survival Analysis
=======================

The objective in `survival analysis`_ (also referred to as time-to-event or reliability analysis)
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

- Python 3.10 or later
- ecos
- joblib
- numexpr
- numpy
- osqp
- pandas 1.4.0 or later
- scikit-learn 1.4 or 1.5
- scipy
- C/C++ compiler

============
Installation
============

The easiest way to install scikit-survival is to use
`Anaconda <https://www.anaconda.com/distribution/>`_ by running::

  conda install -c conda-forge scikit-survival

Alternatively, you can install scikit-survival from source
following `this guide <https://scikit-survival.readthedocs.io/en/stable/install.html#from-source>`_.

========
Examples
========

The `user guide <https://scikit-survival.readthedocs.io/en/stable/user_guide/index.html>`_ provides
in-depth information on the key concepts of scikit-survival, an overview of available survival models,
and hands-on examples in the form of `Jupyter notebooks <https://jupyter.org/>`_.

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

- If you have a question on how to use scikit-survival, please use `GitHub Discussions <https://github.com/sebp/scikit-survival/discussions>`_.
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

Please cite the following paper if you are using **scikit-survival**.

  S. Pölsterl, "scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn,"
  Journal of Machine Learning Research, vol. 21, no. 212, pp. 1–6, 2020.

.. code::

  @article{sksurv,
    author  = {Sebastian P{\"o}lsterl},
    title   = {scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn},
    journal = {Journal of Machine Learning Research},
    year    = {2020},
    volume  = {21},
    number  = {212},
    pages   = {1-6},
    url     = {http://jmlr.org/papers/v21/20-729.html}
  }

.. |License| image:: https://img.shields.io/badge/license-GPLv3-blue.svg
  :target: COPYING
  :alt: License

.. |Codecov| image:: https://codecov.io/gh/sebp/scikit-survival/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/sebp/scikit-survival
  :alt: codecov

.. |Codacy| image:: https://api.codacy.com/project/badge/Grade/17242004cdf6422c9a1052bf1ec63104
   :target: https://app.codacy.com/gh/sebp/scikit-survival/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
   :alt: Codacy Badge

.. |Docs| image:: https://readthedocs.org/projects/scikit-survival/badge/?version=latest
  :target: https://scikit-survival.readthedocs.io/en/latest/
  :alt: readthedocs.org

.. |DOI| image:: https://zenodo.org/badge/77409504.svg
   :target: https://zenodo.org/badge/latestdoi/77409504
   :alt: Digital Object Identifier (DOI)

.. |build-tests| image:: https://github.com/sebp/scikit-survival/actions/workflows/tests-workflow.yaml/badge.svg?branch=master
  :target: https://github.com/sebp/scikit-survival/actions?query=workflow%3Atests+branch%3Amaster
  :alt: GitHub Actions Tests Status

.. |build-windows| image:: https://ci.appveyor.com/api/projects/status/github/sebp/scikit-survival?branch=master&svg=true
   :target: https://ci.appveyor.com/project/sebp/scikit-survival
   :alt: Windows Build Status on AppVeyor

.. _survival analysis: https://en.wikipedia.org/wiki/Survival_analysis
