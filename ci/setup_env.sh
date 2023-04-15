#!/bin/bash
set -xe

OS="$1"

if [ "x${OS}" = "xLinux" ]; then
  COMPILER=()
elif [ "x${OS}" = "xmacOS" ]; then
  COMPILER=(clang_osx-64 clangxx_osx-64)
else
  echo "OS '${OS}' is unsupported."
  exit 1
fi

#conda update -q conda
conda config --set always_yes yes --set changeps1 no
conda config --add pkgs_dirs "${CONDA_PKGS_DIR:-conda_pkgs_dir}"

conda install -n base conda-libmamba-solver

python ci/render-requirements.py ci/deps/requirements.yaml.tmpl > environment.yaml

conda env create -n sksurv-test --solver=libmamba --file environment.yaml

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