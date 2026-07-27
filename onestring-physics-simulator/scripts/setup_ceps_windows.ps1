param(
    [string]$InstallRoot = "C:\CEPS",
    [switch]$ForceReconfigure
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install it and reopen PowerShell."
    }
}

Require-Command git
Require-Command cmake

$source = [System.IO.Path]::GetFullPath($InstallRoot)
$build = Join-Path $source "build"

if (Test-Path -LiteralPath $source) {
    if (-not (Test-Path -LiteralPath (Join-Path $source ".git"))) {
        throw "$source already exists but is not a CEPS git checkout. Choose another -InstallRoot."
    }
    Write-Host "Updating existing CEPS checkout: $source"
    git -C $source fetch origin
    git -C $source checkout main
    git -C $source pull --ff-only origin main
    git -C $source submodule update --init --recursive
} else {
    $parent = Split-Path -Path $source -Parent
    # For a root-level install such as C:\CEPS, $parent is C:\. The drive root
    # already exists and New-Item on it can raise "The path format is invalid"
    # in Windows PowerShell, so create the parent only when it is actually absent.
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Write-Host "Cloning official CEPS into $source"
    git clone --recursive https://github.com/MarkGillespie/CEPS.git $source
}

if ($ForceReconfigure -and (Test-Path -LiteralPath $build)) {
    Remove-Item $build -Recurse -Force
}

Write-Host "Configuring CEPS Release build..."
cmake -S $source -B $build -DCMAKE_BUILD_TYPE=Release

Write-Host "Building parameterize.exe..."
cmake --build $build --config Release --target parameterize --parallel

$candidates = @(
    (Join-Path $build "bin\Release\parameterize.exe"),
    (Join-Path $build "bin\parameterize.exe"),
    (Join-Path $build "Release\parameterize.exe")
)
$executable = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $executable) {
    $executable = Get-ChildItem $build -Filter parameterize.exe -Recurse -File |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $executable) {
    throw "Build completed, but parameterize.exe was not found below $build. Inspect the CMake build output."
}

$executable = [System.IO.Path]::GetFullPath($executable)
$env:ONESTRING_CEPS_EXECUTABLE = $executable
[Environment]::SetEnvironmentVariable(
    "ONESTRING_CEPS_EXECUTABLE",
    $executable,
    [EnvironmentVariableTarget]::User
)

Write-Host ""
Write-Host "Official CEPS executable:" -ForegroundColor Green
Write-Host $executable
Write-Host ""
Write-Host "ONESTRING_CEPS_EXECUTABLE was set for this process and your user account."
Write-Host "Validate from the OneString project with:"
Write-Host "  python scripts\verify_ceps_integration.py"
Write-Host "Then launch:"
Write-Host "  python -m streamlit run app.py"
