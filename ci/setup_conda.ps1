<#
.SYNOPSIS
    Installs and configures Miniforge.
.DESCRIPTION
    This script downloads the Miniforge installer, verifies its integrity,
    and installs it to the specified location. It also initializes conda
    for PowerShell and applies a default configuration.
.PARAMETER DepsVersion
    The version of the dependencies to install.
.PARAMETER InstallationDirectory
    The directory where Miniforge should be installed.
.PARAMETER CondaPkgsDir
    The directory where conda packages should be cached.
#>
[CmdletBinding()]
param(
    [string]$DepsVersion,
    [ValidateNotNullOrEmpty()]
    [string]$InstallationDirectory = "$env:USERPROFILE\Miniforge3",
    [ValidateNotNullOrEmpty()]
    [string]$CondaPkgsDir = $env:CONDA_PKGS_DIR
)

$ErrorActionPreference = "Stop"

function Test-ExitCode {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed with exit code $LASTEXITCODE"
        exit 1
    }
}

$installerUrl = "https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Windows-x86_64.exe"
$installerPath = "Miniforge.exe"
$expectedHash = "b7706a307b005fc397b70a244de19129100906928abccd5592580eb8296fb240"

Write-Verbose "🔽 Downloading Miniforge installer..."
Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

Write-Verbose "🧐 Verifying installer hash..."
$actualHash = (Get-FileHash -Algorithm SHA256 $installerPath).Hash
if ($actualHash -ne $expectedHash) {
    Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value "💥 Hash mismatch! Expected: $expectedHash, Actual: $actualHash"
    exit 1
}

Write-Verbose "::group::🏗️ Installing Miniforge to $InstallationDirectory..."
$installProcess = Start-Process $installerPath -Wait -PassThru -ArgumentList @(
    "/S",
    "/InstallationType=JustMe",
    "/RegisterPython=1",
    "/D=$InstallationDirectory"
)
if ($installProcess.ExitCode -ne 0) {
    Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value "💥 Miniforge installation failed with exit code $($installProcess.ExitCode)"
    exit 1
}
Remove-Item -Path $installerPath
if ($CondaPkgsDir -and -not (Test-Path -Path $CondaPkgsDir)) {
    New-Item -ItemType Directory -Path $CondaPkgsDir -ErrorAction Stop
}
Write-Verbose "::endgroup::"

Write-Verbose "🌐 Updating Path environment variable..."
$env:Path += ";$InstallationDirectory\Scripts;$InstallationDirectory\Library\bin"
Add-Content -Path $env:GITHUB_PATH -Value "$InstallationDirectory\Scripts"
Add-Content -Path $env:GITHUB_PATH -Value "$InstallationDirectory\Library\bin"

Write-Verbose "🔧 Configuring conda..."
conda config --set always_yes yes
Test-ExitCode
conda config --set changeps1 no
Test-ExitCode
conda config --set auto_update_conda false
Test-ExitCode
conda config --set notify_outdated_conda false
Test-ExitCode

Write-Verbose "::group::🎉 Conda installation and configuration complete."
mamba info
Test-ExitCode
Write-Verbose "::endgroup::"

Write-Verbose "::group::✨ Create conda environment..."
$envScript=".\ci\deps\windows\$DepsVersion.ps1"
& $envScript
python "ci\render-requirements.py" "ci\deps\requirements.yaml.tmpl" > environment.yaml

mamba env create -n sksurv-test --file environment.yaml

Add-Content -Path "$InstallationDirectory/envs/sksurv-test/conda-meta/pinned" -Value "numpy $env:CI_NUMPY_VERSION"
Add-Content -Path "$InstallationDirectory/envs/sksurv-test/conda-meta/pinned" -Value "pandas $env:CI_PANDAS_VERSION"
Add-Content -Path "$InstallationDirectory/envs/sksurv-test/conda-meta/pinned" -Value "scikit-learn $env:CI_SKLEARN_VERSION"

mamba list -n sksurv-test
Write-Verbose "::endgroup::"
