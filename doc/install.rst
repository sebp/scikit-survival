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
the *Command Line Tools* package. Open a terminal and execute::

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


.. note::

    If you have not installed the :ref:`dependencies <dependencies>` previously, this command
    will first install all dependencies before installing scikit-survival.
    Therefore, installation might fail if build requirements of some dependencies
    are not met. In particular, `osqp <https://github.com/oxfordcontrol/osqp-python>`_
    does require `CMake <https://cmake.org/>`_ to be installed.

Development Version
^^^^^^^^^^^^^^^^^^^

To install the latest source from our `GitHub repository <https://github.com/sebp/scikit-survival/>`_,
you need to have `Git <https://git-scm.com/>`_ installed and
simply run::

  pip install git+https://github.com/sebp/scikit-survival.git



.. _dependencies:

Dependencies
------------

The current minimum dependencies to run scikit-survival are:

- Python 3.9 or later
- ecos
- joblib
- numexpr
- numpy 1.17.3 or later
- osqp
- pandas 1.0.5 or later
- scikit-learn 1.3
- scipy 1.3.2 or later
- C/C++ compiler
