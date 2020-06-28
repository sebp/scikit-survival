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
from distutils.version import LooseVersion
import os
from pathlib import Path
from pkg_resources import parse_requirements as _parse_requirements
import shutil
import sys

from setuptools import Command, Extension, find_packages, setup

CYTHON_MIN_VERSION = '0.29'


# adapted from bottleneck's setup.py
class clean(Command):
    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self.delete_dirs = []
        self.delete_files = []

        for root, dirs, files in os.walk("sksurv"):
            root = Path(root)
            for d in dirs:
                if d == "__pycache__":
                    self.delete_dirs.append(root / d)

            if "__pycache__" in root.name:
                continue

            for f in map(lambda x: root / x, files):
                ext = f.suffix
                if ext == ".pyc" or ext == ".so":
                    self.delete_files.append(f)

                if ext in (".c", ".cpp",):
                    source_file = f.with_suffix(".pyx")
                    if source_file.exists():
                        self.delete_files.append(f)

        build_path = Path("build")
        if build_path.exists():
            self.delete_dirs.append(build_path)

    def finalize_options(self):
        pass

    def run(self):
        for delete_dir in self.delete_dirs:
            shutil.rmtree(delete_dir)
        for delete_file in self.delete_files:
            delete_file.unlink()


def parse_requirements(filename):
    with open(filename) as fin:
        parsed_requirements = _parse_requirements(
            fin)
        requirements = [str(ir) for ir in parsed_requirements]
    return requirements


def get_long_description():
    with open('README.rst', encoding="utf-8") as fp:
        long_description = fp.read()
    return long_description


EXTENSIONS = {
    "_binarytrees": {"sources": ["sksurv/bintrees/_binarytrees.pyx",
                                 "sksurv/bintrees/binarytrees.cpp"],
                     "language": "c++"},
    "_clinical_kernel": {"sources": ["sksurv/kernels/_clinical_kernel.pyx"]},
    "_coxph_loss": {"sources": ["sksurv/ensemble/_coxph_loss.pyx"]},
    "_prsvm": {"sources": ["sksurv/svm/_prsvm.pyx"]},
    "_minlip": {"sources": ["sksurv/svm/_minlip.pyx"]},
    "_criterion": {"sources": ["sksurv/tree/_criterion.pyx"]},
    "_coxnet": {"sources": ["sksurv/linear_model/_coxnet.pyx"],
                "language": "c++",
                "include_dirs": ["sksurv/linear_model/src",
                                 "sksurv/linear_model/src/eigen"],
                "extra_compile_args": ["-std=c++11"]},
}


def get_module_from_sources(sources):
    for src_path in map(Path, sources):
        if src_path.suffix == ".pyx":
            return ".".join(src_path.parts[:-1] + (src_path.stem,))
    raise ValueError("could not find module from sources: {!r}".format(sources))


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


def cythonize_extensions(extensions):
    """Check that a recent Cython is available and cythonize extensions"""
    _check_cython_version()
    from Cython.Build import cythonize

    # http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments
    directives = {'language_level': '3'}
    cy_cov = os.environ.get('CYTHON_COVERAGE', False)
    if cy_cov:
        directives['linetrace'] = True
        macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]

        for ext in extensions:
            if ext.define_macros is None:
                ext.define_macros = macros
            else:
                ext.define_macros += macros

    return cythonize(extensions,
                     compiler_directives=directives)


def get_extensions():
    import numpy

    numpy_includes = [numpy.get_include()]

    extensions = []
    for config in EXTENSIONS.values():
        name = get_module_from_sources(config["sources"])
        include_dirs = numpy_includes + config.get("include_dirs", [])
        extra_compile_args = config.get("extra_compile_args", [])
        language = config.get("language", "c")
        ext = Extension(
            name=name,
            sources=config["sources"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language=language,
        )
        extensions.append(ext)

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" not in sys.argv and "clean" not in sys.argv:
        extensions = cythonize_extensions(extensions)

    return extensions


def setup_package():
    setup(
        name='scikit-survival',
        url='https://github.com/sebp/scikit-survival',
        project_urls={
            "Bug Tracker": "https://github.com/sebp/scikit-survival/issues",
            "Documentation": "https://scikit-survival.readthedocs.io/en/latest/",
            "Source Code": "https://github.com/sebp/scikit-survival",
        },
        author='Sebastian Pölsterl',
        author_email='sebp@k-d-w.org',
        description='Survival analysis built on top of scikit-learn',
        long_description=get_long_description(),
        license="GPLv3+",
        packages=find_packages(),
        ext_modules=get_extensions(),
        classifiers=['Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                     'Operating System :: MacOS',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Programming Language :: C++',
                     'Programming Language :: Cython',
                     'Programming Language :: Python',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     ],
        zip_safe=False,
        include_package_data=True,
        use_scm_version=True,
        python_requires='>=3.5',
        install_requires=parse_requirements('requirements/prod.txt'),
        cmdclass={"clean": clean},
    )


if __name__ == "__main__":
    setup_package()
