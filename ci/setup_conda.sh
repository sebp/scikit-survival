#!/bin/bash

RUNNER_OS="${1}"
RUNNER_ARCH="${2}"
CONDA_PKGS_DIR="${3}"

run_check_sha() {
    echo "${1}" | shasum -a 256 --check --strict -
}

if [[ "${CONDA:-}" = "" ]]; then
    # download and install conda
    MINICONDA_VERSION="Miniconda3-py312_24.1.2-0"

    if [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "ARM64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-MacOSX-arm64"
        MINICONDA_HASH="1c277b1ec046fd1b628390994e3fa3dbac0e364f44cd98b915daaa67a326c66a"
    elif [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-MacOSX-x86_64"
        MINICONDA_HASH="bc45a2ceea9341579532847cc9f29a9769d60f12e306bba7f0de6ad5acdd73e9"
    elif [[ "${RUNNER_OS}" = "Linux" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-Linux-x86_64"
        MINICONDA_HASH="b978856ec3c826eb495b60e3fffe621f670c101150ebcbdeede4f961f22dc438"
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
