param(
    [string]$Executable = "third_party/affine-body-dynamics/build/Release/abd_sim.exe",
    [string]$SourceDir = "third_party/affine-body-dynamics",
    [string]$OutputDir = "output/abd_official_sample"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = [System.IO.Path]::GetFullPath((Join-Path $root $Executable))
$source = [System.IO.Path]::GetFullPath((Join-Path $root $SourceDir))
$output = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDir))
$scene = Join-Path $source "inputs/cube_drop/cube_drop.json"

if (-not (Test-Path -LiteralPath $exe)) { throw "ABD executable not found: $exe" }
if (-not (Test-Path -LiteralPath $scene)) { throw "Official cube_drop scene not found: $scene" }
New-Item -ItemType Directory -Force -Path $output | Out-Null

& $exe --ngui --scene-path $scene --output-path $output --output-name sim.json --num-steps 5 --log info
if ($LASTEXITCODE -ne 0) { throw "Official Autodesk ABD cube_drop sample failed with exit code $LASTEXITCODE" }

$json = Join-Path $output "sim.json"
$gltf = Join-Path $output "sim.glb"
if (-not (Test-Path -LiteralPath $json)) { throw "ABD did not create sim.json" }
if (-not (Test-Path -LiteralPath $gltf)) { throw "ABD did not create sim.glb" }
$result = Get-Content -Raw -Encoding UTF8 $json | ConvertFrom-Json
if ($result.animation.state_sequence.Count -lt 2) { throw "ABD result contains too few animation states" }
Write-Host "Official Autodesk ABD sample passed: $($result.animation.state_sequence.Count) states"
Write-Host "Set ONESTRING_ABD_EXECUTABLE=$exe"
