#!/bin/bash
set -euo pipefail

RUNNER_OS="${1}"
RUNNER_ARCH="${2}"
CONDA_PKGS_DIR="${3}"

run_check_sha() {
    echo "${1}" | shasum -a 256 --check --strict -
}

if [[ -z "${MINIFORGE:-}" ]]; then
    # download and install conda
    MINIFORGE_VERSION="25.3.1-0"

    if [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "ARM64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-MacOSX-arm64"
        MINIFORGE_HASH="d9eabd1868030589a1d74017b8723b01cf81b5fec1b9da8021b6fa44be7bbeae"
    elif [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-MacOSX-x86_64"
        MINIFORGE_HASH="6c09a3550bb65bdb6d3db6f6c2b890b987b57189f3b71c67a5af49943d2522e8"
    elif [[ "${RUNNER_OS}" = "Linux" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-Linux-x86_64"
        MINIFORGE_HASH="376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768"
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

echo "🌐 Updating Path environment variable..."
export PATH="${MINIFORGE}/bin:${PATH}"
echo "${MINIFORGE}/bin" >> "${GITHUB_PATH}"

echo "::group::🎉 Conda installation and configuration complete."
"${MINIFORGE}/bin/conda" config --show-sources
# Useful for debugging any issues with conda
"${MINIFORGE}/bin/mamba" info
echo "::endgroup::"