param(
    [switch]$InstallCuda,
    [switch]$SkipVisualStudio,
    [switch]$SkipGit,
    [switch]$SkipCMake
)

$ErrorActionPreference = "Stop"

function Install-WingetPackage {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [string[]]$ExtraArgs = @()
    )

    $installed = winget list --id $Id -e --accept-source-agreements 2>$null | Out-String
    if ($LASTEXITCODE -eq 0 -and $installed -match [regex]::Escape($Id)) {
        Write-Host "Already installed: $Id" -ForegroundColor DarkGreen
        return
    }

    Write-Host "Installing: $Id" -ForegroundColor Cyan
    $arguments = @(
        "install", "--id", $Id, "-e",
        "--accept-source-agreements",
        "--accept-package-agreements"
    ) + $ExtraArgs
    & winget @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed while installing $Id (exit code $LASTEXITCODE)"
    }
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "winget was not found. Update/install Microsoft App Installer, then rerun this script."
}

if (-not $SkipGit) {
    Install-WingetPackage -Id "Git.Git"
}
if (-not $SkipCMake) {
    Install-WingetPackage -Id "Kitware.CMake"
}
if (-not $SkipVisualStudio) {
    Install-WingetPackage -Id "Microsoft.VisualStudio.2022.BuildTools" -ExtraArgs @(
        "--override",
        "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    )
}

if ($InstallCuda) {
    # CUDA is optional after this upgrade. The CPU build is the recommended first build.
    $cudaSearch = winget search --id "Nvidia.CUDA" -e --accept-source-agreements 2>$null | Out-String
    if ($LASTEXITCODE -eq 0 -and $cudaSearch -match "Nvidia.CUDA") {
        Install-WingetPackage -Id "Nvidia.CUDA"
    } else {
        throw "The Nvidia.CUDA package was not found in the current winget sources. Install NVIDIA CUDA Toolkit manually, then rerun the build with -EnableCuda."
    }
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    git config --global core.longpaths true
}

Write-Host ""
Write-Host "ABD build environment installation completed." -ForegroundColor Green
Write-Host "Close and reopen PowerShell before building so PATH and Visual Studio discovery are refreshed."
Write-Host "CPU build: powershell -ExecutionPolicy Bypass -File .\scripts\build_autodesk_abd.ps1"
if ($InstallCuda) {
    Write-Host "CUDA build: powershell -ExecutionPolicy Bypass -File .\scripts\build_autodesk_abd.ps1 -EnableCuda -CudaArchitectures 89"
}
