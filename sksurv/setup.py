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

CYTHON_MIN_VERSION = '0.23'


def build_from_c_and_cpp_files(extensions):
    """Modify the extensions to build from the .c and .cpp files.
    This is useful for releases, this way cython is not required to
    run python setup.py install.
    """
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources = sources


def maybe_cythonize_extensions(top_path, config):
    """Tweaks for building extensions between release and development mode."""
    is_release = os.path.exists(os.path.join(top_path, 'PKG-INFO'))

    if is_release:
        build_from_c_and_cpp_files(config.ext_modules)
    else:
        message = ('Please install cython with a version >= {0} in order '
                   'to build a scikit-survival development version.').format(
                       CYTHON_MIN_VERSION)
        try:
            import Cython
            if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
                message += ' Your version of Cython was {0}.'.format(
                    Cython.__version__)
                raise ValueError(message)
            from Cython.Build import cythonize
        except ImportError as exc:
            exc.args += (message,)
            raise

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

    maybe_cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
