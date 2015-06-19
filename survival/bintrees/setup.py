import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('bintrees', parent_package, top_path)

    config.add_extension('_binarytrees',
                         sources=['_binarytrees.cpp', 'binarytrees.cpp'],
                         include_dirs=[numpy.get_include()],
                         language='c++')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
