.. _contributing:

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


.. _forking:

Creating a Fork
---------------

These are the steps you need to take to create a copy of the scikit-survival repository
on your computer.


1. `Create an account <https://github.com/signup>`_ on
   GitHub if you do not already have one.

2. `Fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_
   the `scikit-survival repository <https://github.com/sebp/scikit-survival>`_.

3. Clone your fork of the scikit-survival repository from your GitHub account to your local disk.
   You have to execute from the command line::

    git clone --recurse-submodules git@github.com:YourLogin/scikit-survival.git
    cd scikit-survival


.. _setup-dev-environment:

Setting up a Development Environment
------------------------------------

After you created a copy of our main repository on `GitHub`_, you need
to setup a local development environment.
We strongly recommend to create a separate virtual environment containing all dependencies.

You can use `conda`_ or `uv`_ to create a new virtual environment
and install scikit-survival in development mode.

.. tab-set::
    :sync-group: venv

    .. tab-item:: conda
        :sync: conda

        .. code-block:: bash

            python ci/render-requirements.py ci/deps/requirements.yaml.tmpl > dev-environment.yaml
            conda env create -n sksurv --file dev-environment.yaml
            conda run -n sksurv pip install --group dev -e .

    .. tab-item:: uv
        :sync: uv

        .. code-block:: bash

            uv sync


.. _making-changes-to-code:

Making Changes to the Code
--------------------------
For a pull request to be accepted, your changes must meet the below requirements.

1. All changes related to **one feature** must belong to **one branch**.
   Each branch must be self-contained, with a single new feature or bug fix.
   `Create a new feature branch <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_
   by executing::

    git checkout -b my-new-feature

2. All code must follow the standard Python guidelines for code style,
   `PEP8 <https://peps.python.org/pep-0008/>`_.
   To check that your code conforms to PEP8, you can install
   `tox`_ and run::

    tox -e lint

   Alternatively, you can use `pre-commit`_ to check your code on every commit automatically::

    pre-commit install

3. Each function, class, method, and attribute needs to be documented using doc strings.
   scikit-survival conforms to the
   `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

4. Code submissions must always include unit tests.
   We are using `pytest <https://docs.pytest.org/>`_.
   All tests must be part of the ``tests`` directory.
   You can run the test suite locally by executing::

    pytest

   Tests will also be executed automatically once you submit a pull request.

5. The contributed code will be licensed under the
   `GNU General Public License v3.0 <https://github.com/sebp/scikit-survival/blob/master/COPYING>`_.
   If you did not write the code yourself, you must ensure the existing license
   is compatible and include the license information in the contributed files,
   or obtain a permission from the original author to relicense the contributed code.


.. _submit-pull-request:

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

4. `Create a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.


.. _building-documentation:

Building the Documentation
--------------------------

The documentation resides in the ``doc/`` folder and is written in
reStructuredText. HTML files of the documentation can be generated using `Sphinx`_.
The easiest way to build the documentation is to install `tox`_ and run::

    tox -e docs

Generated files will be located in ``doc/_build/html``. To open the main page
of the documentation, run::

  xdg-open _build/html/index.html


.. _building-cython-code:

Building Cython Code
--------------------

Part of the code base is written in `Cython`_. To rebuild this code after making changes,
please re-run the install command from the :ref:`setup-dev-environment` section.

If you are new to Cython you may find the project's documentation on
:ref:`debugging <cython:debugging>` and :ref:`profiling <cython:profiling>` helpful.

.. _conda: https://conda-forge.org/download/
.. _uv: https://docs.astral.sh/uv/getting-started/installation/
.. _Cython: https://cython.org
.. _Git: https://git-scm.com/
.. _GitHub: https://github.com/sebp/scikit-survival
.. _Sphinx: https://www.sphinx-doc.org/
.. _tox: https://tox.wiki/en/stable/
.. _pre-commit: https://pre-commit.com/#usage
