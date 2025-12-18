#!/bin/bash
set -euo pipefail

python ci/render-requirements.py ci/deps/requirements.yaml.tmpl > environment.yaml

mamba env create -n sksurv-test --file environment.yaml

echo "numpy ${CI_NUMPY_VERSION:?}" > "${MINIFORGE:?}/envs/sksurv-test/conda-meta/pinned"
echo "pandas ${CI_PANDAS_VERSION:?}" >> "${MINIFORGE:?}/envs/sksurv-test/conda-meta/pinned"
echo "scikit-learn ${CI_SKLEARN_VERSION:?}" >> "${MINIFORGE:?}/envs/sksurv-test/conda-meta/pinned"

# delete any version that is already installed.
# use '|| true' to ensure script continues even if package is not found.
mamba run -n sksurv-test pip uninstall --yes scikit-survival || true

mamba list -n sksurv-test
