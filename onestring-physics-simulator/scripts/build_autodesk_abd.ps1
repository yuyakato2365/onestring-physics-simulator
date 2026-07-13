param(
    [string]$SourceDir = "third_party/affine-body-dynamics",
    [string]$BuildDir = "third_party/affine-body-dynamics/build",
    [switch]$EnableCuda,
    [string]$CudaArchitectures = "89",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$source = if ([System.IO.Path]::IsPathRooted($SourceDir)) {
    [System.IO.Path]::GetFullPath($SourceDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $SourceDir))
}
$build = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    [System.IO.Path]::GetFullPath($BuildDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $BuildDir))
}

if (-not (Test-Path -LiteralPath (Join-Path $source "CMakeLists.txt"))) {
    throw "Vendored ABD source was not found: $source"
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake was not found. Run scripts/install_abd_environment.ps1 and reopen PowerShell."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. ABD uses CMake FetchContent to obtain pinned dependencies."
}
if ($EnableCuda -and -not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "-EnableCuda was specified, but nvcc was not found. Install CUDA Toolkit or build without -EnableCuda."
}

if ($Clean -and (Test-Path -LiteralPath $build)) {
    Write-Host "Removing old build directory: $build" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $build
}

New-Item -ItemType Directory -Force -Path $build | Out-Null
git config --global core.longpaths true

$cudaFlag = if ($EnableCuda) { "ON" } else { "OFF" }
$configureArgs = @(
    "-S", $source,
    "-B", $build,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DABD_WITH_OPENGL=OFF",
    "-DABD_WITH_UNIT_TESTS=OFF",
    "-DABD_WITH_TOOLS=OFF",
    "-DABD_WITH_COMPARISONS=OFF",
    "-DABD_WITH_PYTHON=OFF",
    "-DIPC_TOOLKIT_WITH_CUDA=$cudaFlag",
    "-DIPC_TOOLKIT_BUILD_TESTS=OFF",
    "-DPOLYSOLVE_WITH_CHOLMOD=OFF",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
)
if ($EnableCuda) {
    $nvcc = Get-Command nvcc -ErrorAction Stop
    $cudaRoot = Split-Path -Parent (Split-Path -Parent $nvcc.Source)
    $configureArgs += @(
        "-T", "cuda=$cudaRoot",
        "-DCUDAToolkit_ROOT=$cudaRoot",
        "-DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"
    )
    $configureArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

Write-Host "Configuring ABD (CUDA=$cudaFlag)..." -ForegroundColor Cyan
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
}

Write-Host "Building ABD Release..." -ForegroundColor Cyan
& cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) {
    throw "ABD build failed with exit code $LASTEXITCODE"
}

$candidates = @(
    (Join-Path $build "Release/abd_sim.exe"),
    (Join-Path $build "abd_sim.exe")
)
$executable = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $executable) {
    throw "Release build completed, but abd_sim.exe was not found under $build"
}

$help = & $executable --help 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "abd_sim.exe --help failed"
}
if ($help -notmatch "--onestring-manifest") {
    throw "The built executable does not advertise --onestring-manifest. Confirm the upgrade patch was applied before building."
}

$env:ONESTRING_ABD_EXECUTABLE = $executable
Write-Host ""
Write-Host "OneString ABD executable:" -ForegroundColor Green
Write-Host "  $executable"
Write-Host "Current-shell environment variable set: ONESTRING_ABD_EXECUTABLE"
Write-Host "Next: powershell -ExecutionPolicy Bypass -File .\scripts\verify_autodesk_abd.ps1"
