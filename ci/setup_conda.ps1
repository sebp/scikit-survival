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

Write-Host "🏗️ Installing Miniforge to $InstallationDirectory..."
Start-Process $installerPath -Wait -ArgumentList @(
    "/S",
    "/InstallationType=JustMe",
    "/RegisterPython=1",
    "/D=$InstallationDirectory"
)
Remove-Item -Path $installerPath
if ($CondaPkgsDir -and -not (Test-Path -Path $CondaPkgsDir)) {
    New-Item -ItemType Directory -Path $CondaPkgsDir -ErrorAction Stop
}

Write-Host "🌐 Updating Path environment variable..."
$env:Path += ";$InstallationDirectory\Scripts;$InstallationDirectory\Library\bin"
Add-Content -Path $env:GITHUB_PATH -Value "$InstallationDirectory\Scripts"
Add-Content -Path $env:GITHUB_PATH -Value "$InstallationDirectory\Library\bin"

Write-Host "🔧 Configuring conda..."
conda init powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
conda config --set always_yes yes
conda config --set changeps1 no
conda config --set auto_update_conda false
conda config --set notify_outdated_conda false

Write-Host "🎉 Conda installation and configuration complete."
conda info -a
