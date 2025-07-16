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
    MINIFORGE_VERSION="25.3.0-3"

    if [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "ARM64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-MacOSX-arm64"
        MINIFORGE_HASH="16205127ac2b5701881636229b7fe42e1f961007513b8673f8064da331e496a0"
    elif [[ "${RUNNER_OS}" = "macOS" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-MacOSX-x86_64"
        MINIFORGE_HASH="c562e11d8f9caca3dcfb9ba6d5043b9238975d271751e12c3fbfc2a472b4b8fb"
    elif [[ "${RUNNER_OS}" = "Linux" ]] && [[ "${RUNNER_ARCH}" = "X64" ]]; then
        MINIFORGE_FILENAME="${MINIFORGE_VERSION}-Linux-x86_64"
        MINIFORGE_HASH="1b57f8cb991982063f79b56176881093abb1dc76d73fda32102afde60585b5a1"
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
