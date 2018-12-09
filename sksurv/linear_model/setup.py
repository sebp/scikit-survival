from pathlib import Path
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('linear_model', parent_package, top_path)

    eigen_root = Path('src', 'eigen')
    eigen_src = Path(__file__).parent / eigen_root / 'Eigen'
    if not eigen_src.is_dir():
        raise RuntimeError("{} directory not found. You might have to run "
                           "'git submodule update --init'.".format(eigen_src))

    config.add_extension('_coxnet',
                         sources=['_coxnet.pyx'],
                         include_dirs=[numpy.get_include(), 'src',
                                       str(eigen_root)],
                         language='c++',
                         extra_compile_args=["-std=c++11"])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
