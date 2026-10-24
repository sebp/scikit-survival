#!/bin/bash
set -euo pipefail

DEPS_VERSION="${1}"
RUNNER_OS="${2}"
RUNNER_ARCH="${3}"
CONDA_PKGS_DIR="${4}"

run_check_sha() {
    echo "${1}" | shasum -a 256 --check --strict -
}

if [[ -z "${MINIFORGE:-}" ]]; then
    # download and install conda
    MINIFORGE_VERSION="26.9.4-0"

    if [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "ARM64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-MacOSX-arm64"
        MINIFORGE_HASH="85512f5495465644e09655d84bc1df79d978353abd9ef0d0a8f3df75da44c760"
    elif [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-MacOSX-x86_64"
        MINIFORGE_HASH="4bf8d162f90a9ae79fbe1c3b152a9957b6cf4e7385758fa21e0340229d444df1"
    elif [[ "${RUNNER_OS}" = "Linux" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-Linux-x86_64"
        MINIFORGE_HASH="621428f40496a90702753b09237794518a081d9979d9f845a525de761b6056f7"
    else
        echo "Unsupported OS or ARCH: ${RUNNER_OS} ${RUNNER_ARCH}"
        exit 1
    fi

    export MINIFORGE="${GITHUB_WORKSPACE}/miniforge"

    echo "::group::🔽 Downloading Miniforge installer..."
    mkdir -p "${MINIFORGE}"
    curl --fail -L \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_FILENAME}.sh" \
        -o "${MINIFORGE}/miniforge.sh"
    echo "::endgroup::"

    echo "::group::🧐 Verifying installer hash..."
    run_check_sha "${MINIFORGE_HASH}  ${MINIFORGE}/miniforge.sh"
    echo "::endgroup::"

    echo "::group::🏗️ Installing Miniforge to ${MINIFORGE}..."
    bash "${MINIFORGE}/miniforge.sh" -b -u -p "${MINIFORGE}"
    rm -rf "${MINIFORGE}/miniforge.sh"
    echo "::endgroup::"

    echo "MINIFORGE=${MINIFORGE}" >> "${GITHUB_ENV}"
fi

# The directory in which packages are located.
# https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/settings.html#pkgs-dirs-specify-package-directories
# Note: sudo chown is unnecessary in GitHub Actions environments as the runner user has permissions.
if [[ ! -d "${CONDA_PKGS_DIR}" ]]; then
    mkdir -p "${CONDA_PKGS_DIR}"
fi

# Configure conda
echo "🔧 Configuring conda..."
"${MINIFORGE}/bin/conda" config --set always_yes yes
"${MINIFORGE}/bin/conda" config --set changeps1 no
"${MINIFORGE}/bin/conda" config --set auto_update_conda false
"${MINIFORGE}/bin/conda" config --set show_channel_urls true

# check whether the script is executed on GitHub Actions
if [[ -n "${CI:-}" ]]; then
echo "🌐 Updating Path environment variable..."
export PATH="${MINIFORGE}/bin:${PATH}"
echo "${MINIFORGE}/bin" >> "${GITHUB_PATH}"
fi

echo "::group::🎉 Conda installation and configuration complete."
"${MINIFORGE}/bin/conda" config --show-sources
# Useful for debugging any issues with conda
"${MINIFORGE}/bin/mamba" info
echo "::endgroup::"

echo "::group::✨ Create conda environment..."
source "ci/deps/${DEPS_VERSION:?}.sh"
python ci/render-requirements.py ci/deps/requirements.yaml.tmpl > environment.yaml

mamba env create -n sksurv-test --file environment.yaml

echo "numpy ${CI_NUMPY_VERSION:?}" > "${MINIFORGE:?}/envs/sksurv-test/conda-meta/pinned"
echo "pandas ${CI_PANDAS_VERSION:?}" >> "${MINIFORGE:?}/envs/sksurv-test/conda-meta/pinned"
echo "scikit-learn ${CI_SKLEARN_VERSION:?}" >> "${MINIFORGE:?}/envs/sksurv-test/conda-meta/pinned"

# delete any version that is already installed.
# use '|| true' to ensure script continues even if package is not found.
mamba run -n sksurv-test pip uninstall --yes scikit-survival || true

mamba list -n sksurv-test
echo "::endgroup::"
