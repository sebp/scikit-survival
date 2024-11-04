#!/bin/bash

RUNNER_OS="${1}"
RUNNER_ARCH="${2}"
CONDA_PKGS_DIR="${3}"

run_check_sha() {
    echo "${1}" | shasum -a 256 --check --strict -
}

if [[ "${CONDA:-}" = "" ]]; then
    # download and install conda
    MINICONDA_VERSION="Miniconda3-py312_24.9.2-0"

    if [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "ARM64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-MacOSX-arm64"
        MINICONDA_HASH="08d8a82ed21d2dae707554d540b172fe03327347db747644fbb33abfaf07fddd"
    elif [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-MacOSX-x86_64"
        MINICONDA_HASH="ce3b440c32c9c636bbe529477fd496798c35b96d9db1838e3df6b0a80714da4e"
    elif [[ "${RUNNER_OS}" = "Linux" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINICONDA_VERSION="${MINICONDA_VERSION}-Linux-x86_64"
        MINICONDA_HASH="8d936ba600300e08eca3d874dee88c61c6f39303597b2b66baee54af4f7b4122"
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
