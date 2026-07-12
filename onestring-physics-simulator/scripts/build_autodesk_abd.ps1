param(
    [string]$SourceDir = "third_party/affine-body-dynamics",
    [string]$BuildDir = "third_party/affine-body-dynamics/build",
    [string]$CudaArchitectures = "75"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$source = [System.IO.Path]::GetFullPath((Join-Path $root $SourceDir))
$build = [System.IO.Path]::GetFullPath((Join-Path $root $BuildDir))

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake is required. Install CMake and a Visual Studio C++ workload before building Autodesk ABD."
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "CUDA nvcc is required by the current Autodesk ABD CMake project (LANGUAGES CXX CUDA)."
}
if (-not (Test-Path -LiteralPath (Join-Path $source "CMakeLists.txt"))) {
    git clone https://github.com/Autodesk/affine-body-dynamics.git $source
    git -c safe.directory='*' -C $source config core.longpaths true
}

cmake -S $source -B $build `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures `
    -DABD_WITH_OPENGL=OFF `
    -DABD_WITH_UNIT_TESTS=ON `
    -DABD_WITH_TOOLS=ON
cmake --build $build --config Release --parallel

$candidate = Join-Path $build "Release/abd_sim.exe"
if (-not (Test-Path -LiteralPath $candidate)) {
    $candidate = Join-Path $build "abd_sim.exe"
}
if (-not (Test-Path -LiteralPath $candidate)) {
    throw "Release build completed but abd_sim.exe was not found under $build"
}
Write-Host "Autodesk ABD Release executable: $candidate"
