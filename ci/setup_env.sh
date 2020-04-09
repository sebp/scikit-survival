#!/bin/bash
set -xe

conda config --set always_yes yes --set changeps1 no

conda create -n sksurv-test \
  python=$CONDA_PYTHON_VERSION \
  numpy=$NUMPY_VERSION \
  pandas=$PANDAS_VERSION \
  scikit-learn=$SKLEARN_VERSION \
  gcc_linux-64 \
  gxx_linux-64

echo "numpy $NUMPY_VERSION.*" > "$CONDA/envs/sksurv-test/conda-meta/pinned"
echo "pandas $PANDAS_VERSION.*" >> "$CONDA/envs/sksurv-test/conda-meta/pinned"
echo "scikit-learn $SKLEARN_VERSION.*" >> "$CONDA/envs/sksurv-test/conda-meta/pinned"

# Useful for debugging any issues with conda
conda info -a

source activate sksurv-test

# delete any version that is already installed
pip uninstall --yes scikit-survival || exit 0