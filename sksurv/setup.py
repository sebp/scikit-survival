# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os
import os.path
from distutils.version import LooseVersion
import sys

CYTHON_MIN_VERSION = '0.29'


def _check_cython_version():
    message = ("Please install Cython with a version >= {0} in order "
               "to build a scikit-learn from source.").format(
                   CYTHON_MIN_VERSION)
    try:
        import Cython
    except ModuleNotFoundError:
        # Re-raise with more informative error message instead:
        raise ModuleNotFoundError(message)

    if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
        message += (" The current version of Cython is {} installed in {}."
                    .format(Cython.__version__, Cython.__path__))
        raise ValueError(message)


def cythonize_extensions(top_path, config):
    """Check that a recent Cython is available and cythonize extensions"""
    _check_cython_version()
    from Cython.Build import cythonize

    # http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments
    directives = {'language_level': '3'}
    cy_cov = os.environ.get('CYTHON_COVERAGE', False)
    if cy_cov:
        directives['linetrace'] = True
        macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
    else:
        macros = []

    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives=directives)

    for e in config.ext_modules:
        e.define_macros.extend(macros)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('sksurv', parent_package, top_path)
    config.add_subpackage('bintrees')
    config.add_subpackage('datasets')
    config.add_subpackage('ensemble')
    config.add_subpackage('io')
    config.add_subpackage('kernels')
    config.add_subpackage('linear_model')
    config.add_subpackage('meta')
    config.add_subpackage('svm')

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
