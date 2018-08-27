import os
import os.path
import sys
from pkg_resources import parse_requirements as _parse_requirements

from distutils.command.sdist import sdist
from setuptools import find_packages


def parse_requirements(filename):
    with open(filename) as fin:
        parsed_requirements = _parse_requirements(
            fin)
        requirements = [str(ir) for ir in parsed_requirements]
    return requirements


with open('README.rst') as fp:
    long_description = fp.read()


requirements = parse_requirements('requirements/prod.txt')


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('sksurv')

    return config


def setup_package():
    metadata = dict(name='scikit-survival',
                    url='https://github.com/sebp/scikit-survival',
                    author='Sebastian Pölsterl',
                    author_email='sebp@k-d-w.org',
                    description='Survival analysis built on top of scikit-learn',
                    long_description=long_description,
                    license="GPLv3+",
                    packages=find_packages(),
                    classifiers=['Development Status :: 4 - Beta',
                                 'Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                    ],
                    zip_safe=False,
                    include_package_data=True,
                    use_scm_version=True,
                    setup_requires=['setuptools_scm'],
                    install_requires=requirements,
                    cmdclass={'sdist': sdist},
    )

    if (len(sys.argv) >= 2
        and ('--help' in sys.argv[1:] or sys.argv[1]
        in ('--help-commands', 'egg_info', '--version', 'clean'))):

        # For these actions, NumPy is not required.
        from setuptools import setup
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    py_version = sys.version_info[:2]
    if py_version < (3, 5):
        raise RuntimeError('Python 3.5 or later is required')

    setup_package()
