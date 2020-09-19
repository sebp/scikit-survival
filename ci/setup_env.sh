#!/bin/bash
set -xe

OS="$1"

if [[ "x${OS}" = "xLinux" ]]; then
  COMPILER="gcc_linux-64 gxx_linux-64"
elif [[ "x${OS}" = "xmacOS" ]]; then
  COMPILER="clang_osx-64 clangxx_osx-64"
else
  echo "OS '${OS}' is unsupported."
  exit 1
fi

conda config --set always_yes yes --set changeps1 no

conda create -n sksurv-test \
  python=$CONDA_PYTHON_VERSION \
  numpy=$NUMPY_VERSION \
  pandas=$PANDAS_VERSION \
  scikit-learn=$SKLEARN_VERSION \
  $COMPILER

echo "numpy $NUMPY_VERSION.*" > "$CONDA/envs/sksurv-test/conda-meta/pinned"
echo "pandas $PANDAS_VERSION.*" >> "$CONDA/envs/sksurv-test/conda-meta/pinned"
echo "scikit-learn $SKLEARN_VERSION.*" >> "$CONDA/envs/sksurv-test/conda-meta/pinned"

# Useful for debugging any issues with conda
conda info -a

source activate sksurv-test

# delete any version that is already installed
pip uninstall --yes scikit-survival || exit 0