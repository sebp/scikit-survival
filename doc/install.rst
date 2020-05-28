Installing scikit-survival
==========================

This is the recommended and easiest to install scikit-survival is to use
:ref:`install-conda`.
Alternatively, you can install scikit-survival :ref:`install-from-source`.

.. _install-conda:

Anaconda
--------

Pre-built binary packages for Linux, MacOS, and Windows are available
for `Anaconda <https://www.anaconda.com/distribution/>`_.
If you have Anaconda installed, run::

  conda install -c sebp scikit-survival


.. _install-from-source:

From Source
-----------

If you want to build scikit-survival from source, you
will need a C/C++ compiler to compile extensions.

**Linux**

On Linux, you need to install *gcc*, which in most cases is available
via your distribution's packaging system.
Please follow your distribution's instructions on how to install packages.

**MacOS**

On MacOS, you need to install *clang*, which is available from
the *Command Line Tools* package. Open a terminal and excecute::

  xcode-select --install

Alternatively, you can download it from the
`Apple Developers page <https://developer.apple.com/downloads/index.action>`_.
Log in with your Apple ID, then search and download the
*Command Line Tools for Xcode* package.

**Windows**

On Windows, the compiler you need depends on the Python version
you are using. See `this guide <https://wiki.python.org/moin/WindowsCompilers>`_
to determine which Microsoft Visual C++ compiler to use with a specific Python version.


Latest Release
^^^^^^^^^^^^^^

To install the latest release of scikit-survival from source, run::

  pip install scikit-survival


Development Version
^^^^^^^^^^^^^^^^^^^

To install the latest source from our `GitHub repository <https://github.com/sebp/scikit-survival/>`_,
you need to have `Git <https://git-scm.com/>`_ installed and
simply run::

  pip install git+https://github.com/sebp/scikit-survival.git


Dependencies
------------

The current minimum dependencies to run scikit-survival are:

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