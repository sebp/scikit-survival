Contributing Guidelines
=======================

This page explains how you can contribute to the development of scikit-survival.
There are a lot of ways you can contribute:

- Writing new code, e.g. implementations of new algorithms, or examples.
- Fixing bugs.
- Improving documentation.
- Reviewing open pull requests.

scikit-survival is developed on `GitHub`_ using the `Git`_ version control system.
The preferred way to contribute to scikit-survival is to fork
the main repository on GitHub, then submit a *pull request* (PR).

Creating a fork
---------------

These are the steps you need to take to create a copy of the scikit-survival repository
on your computer.


1. `Create an account <https://github.com/join>`_ on
   GitHub if you do not already have one.

2. `Fork <https://help.github.com/en/github/getting-started-with-github/fork-a-repo>`_
   the `scikit-survival repository <https://github.com/sebp/scikit-survival>`_.

3. Clone your fork of the scikit-survival repository from your GitHub account to your local disk.
   You have to execute from the command line::

    git clone --recurse-submodules git@github.com:YourLogin/scikit-survival.git
    cd scikit-survival


Setting up a Development Environment
------------------------------------

After you created a copy of our main repository on `GitHub`_, your need
to setup a local development environment.
We strongly recommend to use `conda`_ to
create a separate virtual environment containing all dependencies.
These are the steps you need to take.

1. Install `conda`_ for your operating system if you haven't already.

2. Create a new environment, named ``sksurv``::

    python ci/list-requirements.py requirements/dev.txt > dev-requirements.txt
    conda create -n sksurv -c sebp python=3 --file dev-requirements.txt


3. Activate the newly created environment::

    conda activate sksurv

4. Compile the C/C++ extensions and install scikit-survival in development mode::

    pip install --no-build-isolation -e .

Making Changes to the Code
--------------------------
For a pull request to be accepted, your changes must meet the below requirements.

1. All changes related to **one feature** must belong to **one branch**.
   Each branch must be self-contained, with a single new feature or bugfix.
   `Create a new feature branch <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_
   by executing::

    git checkout -b my-new-feature

2. All code must follow the standard Python guidelines for code style,
   `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
   To check that your code conforms to PEP8, you can install
   `tox`_ and run::

    tox -e flake8

3. Each function, class, method, and attribute needs to be documented using docstrings.
   scikit-survival conforms to the
   `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

4. Code submissions must always include unit tests.
   We are using `pytest <https://docs.pytest.org/>`_.
   All tests must be part of the ``tests`` directory.
   You can run the test suite locally by executing::

    py.test tests/

   Tests will also be executed automatically once you submit a pull request.

5. The contributed code will be licensed under the
   `GNU General Public License v3.0 <https://github.com/sebp/scikit-survival/blob/master/COPYING>`_.
   If you did not write the code yourself, you must ensure the existing license
   is compatible and include the license information in the contributed files,
   or obtain a permission from the original author to relicense the contributed code.

Submitting a Pull Request
-------------------------

1. When you are done coding in your feature branch,
   `add changed or new files <https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_tracking_files>`_::

    git add path/to/modified_file

2. Create a `commit <https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_committing_changes>`_
   message describing what you changed. Commit messages should be clear and concise.
   The first line should contain the subject of the commit and not exceed 80 characters
   in length. If necessary, add a blank line after the subject followed by a commit message body
   with more details::

    git commit

3. Push the changes to GitHub::

    git push -u origin my_feature

4. `Create a pull request <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`_.


Building the Documentation
--------------------------

The documentation resides in the ``doc/`` folder and is written in
reStructuredText. HTML files of the documentation can be generated using `Sphinx`_.
The easiest way to build the documentation is to install `tox`_ and run::

    tox -e docs

Generated files will be located in ``doc/_build/html``. To open the main page
of the documentation, run::

  xdg-open _build/html/index.html

.. _conda: https://conda.io/miniconda.html
.. _Git: https://git-scm.com/
.. _GitHub: https://github.com/sebp/scikit-survival
.. _Sphinx: https://www.sphinx-doc.org/
.. _tox: https://tox.readthedocs.io/en/latest/