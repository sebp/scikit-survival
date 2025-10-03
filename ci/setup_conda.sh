#!/bin/bash
set -x

RUNNER_OS="${1}"
RUNNER_ARCH="${2}"
CONDA_PKGS_DIR="${3}"

run_check_sha() {
    echo "${1}" | shasum -a 256 --check --strict -
}

if [[ "${MINIFORGE:-}" = "" ]]; then
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

    mkdir -p "${MINIFORGE}" && \
    curl --fail -L "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_FILENAME}.sh" -o "${MINIFORGE}/miniforge.sh" && \
    run_check_sha "${MINIFORGE_HASH}  ${MINIFORGE}/miniforge.sh" && \
    bash "${MINIFORGE}/miniforge.sh" -b -u -p "${MINIFORGE}" && \
    rm -rf "${MINIFORGE}/miniforge.sh" || exit 1

    echo "MINIFORGE=${MINIFORGE}" >> "${GITHUB_ENV}"
fi

"${MINIFORGE}/bin/conda" config --set always_yes yes && \
"${MINIFORGE}/bin/conda" config --set changeps1 no && \
"${MINIFORGE}/bin/conda" config --set auto_update_conda false && \
"${MINIFORGE}/bin/conda" config --set show_channel_urls true || \
exit 1

"${MINIFORGE}/bin/conda" config --show-sources

# The directory in which packages are located.
# https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/settings.html#pkgs-dirs-specify-package-directories
if [[ ! -d "${CONDA_PKGS_DIR}" ]]; then
    mkdir -p "${CONDA_PKGS_DIR}" || exit 1
fi
sudo chown -R "${USER}" "${CONDA_PKGS_DIR}" || \
exit 1

sudo "${MINIFORGE}/bin/conda" update -q -n base conda && \
sudo chown -R "${USER}" "${MINIFORGE}" || \
exit 1

export PATH="${MINIFORGE}/bin:${PATH}"
echo "${MINIFORGE}/bin" >> "${GITHUB_PATH}"
