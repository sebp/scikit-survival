Installing scikit-survival
==========================

The recommended and easiest way to install scikit-survival is to use
:ref:`install-conda` or :ref:`install-pip`.
Pre-built binary packages for scikit-survival are available for Linux, macOS, and Windows.
Alternatively, you can install scikit-survival :ref:`install-from-source`.

.. _install-conda:

conda
-----

If you have `conda <https://docs.anaconda.com/>`_ installed, you can
install scikit-survival from the ``conda-forge`` channel by running::

  conda install -c conda-forge scikit-survival

.. _install-pip:

pip
---

If you use ``pip``, install the latest release of scikit-survival with::

  pip install scikit-survival


.. _install-from-source:

From Source
-----------

If you want to build scikit-survival from source, you
will need a C/C++ compiler to compile extensions.

**Linux**

On Linux, you need to install *gcc*, which in most cases is available
via your distribution's packaging system.
Please follow your distribution's instructions on how to install packages.

**macOS**

On macOS, you need to install *clang*, which is available from
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

  pip install scikit-survival --no-binary scikit-survival


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
