#!/bin/bash
set -xe

MINICONDA_DIR="$HOME/miniconda3"

if [ -d "$MINICONDA_DIR" ] && [ -e "$MINICONDA_DIR/bin/conda" ]
then
  echo "Miniconda install already present from cache: $MINICONDA_DIR"
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
else
  echo "Installing Miniconda"
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b -f -p "$MINICONDA_DIR"

  conda config --set always_yes yes --set changeps1 no
  conda config --set auto_update_conda false
  conda update -q conda

  conda create -n sksurv-test \
    python=$CONDA_PYTHON_VERSION \
    gcc_linux-64 \
    gxx_linux-64

  conda install -n sksurv-test -c conda-forge \
    numpy=$NUMPY_VERSION \
    pandas=$PANDAS_VERSION \
    scikit-learn=$SKLEARN_VERSION \
    cython \
    tox \
    "blas=*=openblas"
  echo "numpy $NUMPY_VERSION.*" > "$MINICONDA_DIR/envs/sksurv-test/conda-meta/pinned"
  echo "pandas $PANDAS_VERSION.*" >> "$MINICONDA_DIR/envs/sksurv-test/conda-meta/pinned"
  echo "scikit-learn $SKLEARN_VERSION.*" >> "$MINICONDA_DIR/envs/sksurv-test/conda-meta/pinned"
fi

# Useful for debugging any issues with conda
conda info -a

source activate sksurv-test

# delete any version that is already installed
pip uninstall --yes scikit-survival || exit 0