#!/bin/bash
set -xe

OS="$1"

if [ "x${OS}" = "xLinux" ]; then
  COMPILER=()
elif [ "x${OS}" = "xmacOS" ]; then
  COMPILER=(clang_osx-arm64 clangxx_osx-arm64)
else
  echo "OS '${OS}' is unsupported."
  exit 1
fi

python ci/render-requirements.py ci/deps/requirements.yaml.tmpl > environment.yaml

conda env create -n sksurv-test --file environment.yaml

echo "numpy ${CI_NUMPY_VERSION:?}" > "${CONDA:?}/envs/sksurv-test/conda-meta/pinned"
echo "pandas ${CI_PANDAS_VERSION:?}" >> "${CONDA:?}/envs/sksurv-test/conda-meta/pinned"
echo "scikit-learn ${CI_SKLEARN_VERSION:?}" >> "${CONDA:?}/envs/sksurv-test/conda-meta/pinned"

# Useful for debugging any issues with conda
conda info -a

# shellcheck disable=SC1091
source activate sksurv-test

# delete any version that is already installed
pip uninstall --yes scikit-survival || exit 0

conda list -n sksurv-test