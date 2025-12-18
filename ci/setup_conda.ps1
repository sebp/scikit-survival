<#
.SYNOPSIS
    Installs and configures Miniforge.
.DESCRIPTION
    This script downloads the Miniforge installer, verifies its integrity,
    and installs it to the specified location. It also initializes conda
    for PowerShell and applies a default configuration.
.PARAMETER InstallationDirectory
    The directory where Miniforge should be installed.
.PARAMETER CondaPkgsDir
    The directory where conda packages should be cached.
#>
param(
    [string]$InstallationDirectory = "$env:USERPROFILE\Miniforge3",
    [string]$CondaPkgsDir = $env:CONDA_PKGS_DIR
)

$ErrorActionPreference = "Stop"

function Check-ExitCode {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed with exit code $LASTEXITCODE"
        exit 1
    }
}

$installerUrl = "https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Windows-x86_64.exe"
$installerPath = "Miniforge.exe"
$expectedHash = "b7706a307b005fc397b70a244de19129100906928abccd5592580eb8296fb240"

Write-Host "🔽 Downloading Miniforge installer..."
Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

Write-Host "🧐 Verifying installer hash..."
$actualHash = (Get-FileHash -Algorithm SHA256 $installerPath).Hash
if ($actualHash -ne $expectedHash) {
    Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value "💥 Hash mismatch! Expected: $expectedHash, Actual: $actualHash"
    exit 1
}

Write-Host "::group::🏗️ Installing Miniforge to $InstallationDirectory..."
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
Write-Host "::endgroup::"

Write-Host "🌐 Updating Path environment variable..."
$env:Path += ";$InstallationDirectory\Scripts;$InstallationDirectory\Library\bin"
Add-Content -Path $env:GITHUB_PATH -Value "$InstallationDirectory\Scripts"
Add-Content -Path $env:GITHUB_PATH -Value "$InstallationDirectory\Library\bin"

Write-Host "🔧 Configuring conda..."
conda config --set always_yes yes
Check-ExitCode
conda config --set changeps1 no
Check-ExitCode
conda config --set auto_update_conda false
Check-ExitCode
conda config --set notify_outdated_conda false
Check-ExitCode

Write-Host "::group::🎉 Conda installation and configuration complete."
mamba info
Check-ExitCode
Write-Host "::endgroup::"
