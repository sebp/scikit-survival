from os.path import join
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('linear_model', parent_package, top_path)

    config.add_extension('_coxnet',
                         sources=['_coxnet.pyx'],
                         include_dirs=[numpy.get_include(), 'src',
                                       join('src', 'eigen')],
                         language='c++',
                         extra_compile_args=["-std=c++11"])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
