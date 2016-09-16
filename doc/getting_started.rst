Getting Started
===============

Requirements
------------

- Python 3.4 or later
- cvxpy
- cvxopt
- numexpr
- numpy 1.9 or later
- pandas 0.17.0 or later
- scikit-learn 0.17
- scipy 0.16 or later
- C/C++ compiler


Installation
------------

The easiest way to get started is to install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
and setup an environment. To create a new environment, named `ssvm`, run::

  conda create -n ssvm python=3 --file requirements.txt

To work in this environment, ``activate`` it as follows::

  source activate ssvm

If you are on Windows, run the above command without the ``source`` in the beginning.

Once you setup your build environment, you have to compile the C/C++
extensions and install the package by running::

  python setup.py install

Alternatively, if you want to use the package without installing it,
you can compile the extensions in place by running::

  python setup.py build_ext --inplace

To check everything is setup correctly run the test suite by executing::

  nosetests
