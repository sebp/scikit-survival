#!/bin/bash

RUNNER_OS="${1}"
RUNNER_ARCH="${2}"
CONDA_PKGS_DIR="${3}"

run_check_sha() {
    echo "${1}" | shasum -a 256 --check --strict -
}

if [[ "${CONDA:-}" = "" ]]; then
    # download and install conda
    MINICONDA_VERSION="Miniconda3-py313_25.3.1-1"

    if [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "ARM64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-MacOSX-arm64"
        MINICONDA_HASH="d54b27ed4a6d3c31fedbad6f9f488377702196b0d8d89854e8e7d01f701f225b"
    elif [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-MacOSX-x86_64"
        MINICONDA_HASH="614c455b74d85abe98c2d0fb9b00628bbf2d48932ea4b49ec05b5c4bee7e9239"
    elif [[ "${RUNNER_OS}" = "Linux" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-Linux-x86_64"
        MINICONDA_HASH="53a86109463cfd70ba7acab396d416e623012914eee004729e1ecd6fe94e8c69"
    else
        echo "Unsupported OS or ARCH: ${RUNNER_OS} ${RUNNER_ARCH}"
        exit 1
    fi

    export CONDA="${GITHUB_WORKSPACE}/miniconda3"

    mkdir -p "${CONDA}" && \
    curl "https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh" -o "${CONDA}/miniconda.sh" && \
    run_check_sha "${MINICONDA_HASH}  ${CONDA}/miniconda.sh" && \
    bash "${CONDA}/miniconda.sh" -b -u -p "${CONDA}" && \
    rm -rf "${CONDA}/miniconda.sh" || exit 1

    echo "CONDA=${CONDA}" >> "${GITHUB_ENV}"
fi

"${CONDA}/bin/conda" config --set always_yes yes && \
"${CONDA}/bin/conda" config --set changeps1 no && \
"${CONDA}/bin/conda" config --set auto_update_conda false && \
"${CONDA}/bin/conda" config --set show_channel_urls true || \
exit 1

# The directory in which packages are located.
# https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/settings.html#pkgs-dirs-specify-package-directories
if [[ ! -d "${CONDA_PKGS_DIR}" ]]; then
    mkdir -p "${CONDA_PKGS_DIR}" || exit 1
fi
sudo chown -R "${USER}" "${CONDA_PKGS_DIR}" || \
exit 1

sudo "${CONDA}/bin/conda" update -q -n base conda && \
sudo chown -R "${USER}" "${CONDA}" || \
exit 1

export PATH="${CONDA}/bin:${PATH}"
echo "${CONDA}/bin" >> "${GITHUB_PATH}"
