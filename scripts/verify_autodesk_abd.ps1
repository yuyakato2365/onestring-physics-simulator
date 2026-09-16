param(
    [string]$Executable = "third_party/affine-body-dynamics/build/Release/abd_sim.exe",
    [string]$OutputDir = "output/abd_onestring_smoke",
    [int]$Steps = 20
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = if ([System.IO.Path]::IsPathRooted($Executable)) {
    [System.IO.Path]::GetFullPath($Executable)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $Executable))
}
if (-not (Test-Path -LiteralPath $exe)) {
    $fallback = [System.IO.Path]::GetFullPath(
        (Join-Path $root "third_party/affine-body-dynamics/build/abd_sim.exe")
    )
    if (Test-Path -LiteralPath $fallback) {
        $exe = $fallback
    }
}
if (-not (Test-Path -LiteralPath $exe)) {
    throw "ABD executable not found: $exe"
}

$help = & $exe --help 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "abd_sim.exe --help failed" }
if ($help -notmatch "--onestring-manifest") {
    throw "This executable does not include the OneString extension (--onestring-manifest missing)."
}

$output = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDir))
$assets = Join-Path $output "assets"
$resultDir = Join-Path $output "result"
Remove-Item -Recurse -Force $output -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $assets, $resultDir | Out-Null

$cubeObj = @"
v -0.2 -0.2 -0.2
v  0.2 -0.2 -0.2
v  0.2  0.2 -0.2
v -0.2  0.2 -0.2
v -0.2 -0.2  0.2
v  0.2 -0.2  0.2
v  0.2  0.2  0.2
v -0.2  0.2  0.2
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"@
$mesh0 = Join-Path $assets "body_0000.obj"
$mesh1 = Join-Path $assets "body_0001.obj"
Set-Content -LiteralPath $mesh0 -Value $cubeObj -Encoding ASCII
Set-Content -LiteralPath $mesh1 -Value $cubeObj -Encoding ASCII

$scene = [ordered]@{
    scene_type = "distance_barrier_rb_problem"
    solver = "ipc_solver"
    timestep = 0.01
    max_iterations = $Steps
    distance_barrier_constraint = [ordered]@{
        trajectory_type = "ACCD"
        initial_barrier_activation_distance = 0.001
        minimum_separation_distance = 0.000001
    }
    ipc_solver = [ordered]@{
        velocity_conv_tol = 0.001
    }
    friction_constraints = [ordered]@{
        iterations = 0
    }
    rigid_body_problem = [ordered]@{
        coefficient_restitution = -1.0
        coefficient_friction = 0.0
        gravity = @(0.0, 0.0, 0.0)
        orthogonality_stiffness = 1000000000.0
        do_intersection_check = $true
        rigid_bodies = @(
            [ordered]@{
                mesh = $mesh0
                position = @(0.0, 0.0, 0.0)
                rotation = @(0.0, 0.0, 0.0)
                density = 1000.0
                oriented = $true
                type = "dynamic"
                group_id = 0
            },
            [ordered]@{
                mesh = $mesh1
                position = @(2.5, 0.0, 0.0)
                rotation = @(0.0, 0.0, 0.0)
                density = 1000.0
                oriented = $true
                type = "dynamic"
                group_id = 1
            }
        )
        linear_constraints = @()
    }
}

$duration = 0.01 * $Steps
$manifest = [ordered]@{
    schema = "onestring-abd-bridge-v2"
    tile_count = 2
    string = [ordered]@{
        model = "unilateral_total_path_length_constraint"
        path_points = @(
            [ordered]@{
                type = "world_anchor"
                id = "support"
                position = @(0.0, 0.0, 0.0)
            },
            [ordered]@{
                type = "body_guide"
                body_id = 0
                material_point = @(0.0, 0.0, 0.0)
            },
            [ordered]@{
                type = "body_guide"
                body_id = 1
                material_point = @(0.0, 0.0, 0.0)
            }
        )
        initial_length = 2.5
        stiffness = 1000000.0
        smoothing_epsilon = 0.000000001
        use_exact_hessian = $false
        pull_schedule = @(
            [ordered]@{ time = 0.0; command_length = 2.5 },
            [ordered]@{ time = $duration; command_length = 1.8 }
        )
        inequality = "L(q) <= L_command(t)"
        compression_force_when_slack = 0.0
    }
    shake_trajectory = [ordered]@{
        amplitude = 0.0
        frequency_hz = 0.0
        direction = @(1.0, 0.0, 0.0)
        start_time = 0.0
        end_time = 0.0
        target_anchor = "support"
    }
}

$scenePath = Join-Path $output "scene.json"
$manifestPath = Join-Path $output "onestring_manifest.json"
$scene | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $scenePath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Running self-contained OneString ABD smoke scene..." -ForegroundColor Cyan
& $exe `
    --ngui `
    --scene-path $scenePath `
    --onestring-manifest $manifestPath `
    --output-path $resultDir `
    --output-name sim.json `
    --num-steps $Steps `
    --log info
if ($LASTEXITCODE -ne 0) {
    throw "OneString ABD smoke run failed with exit code $LASTEXITCODE"
}

$resultPath = Join-Path $resultDir "sim.json"
$gltfPath = Join-Path $resultDir "sim.glb"
if (-not (Test-Path -LiteralPath $resultPath)) {
    throw "ABD did not create sim.json"
}
if (-not (Test-Path -LiteralPath $gltfPath)) {
    throw "ABD did not create sim.glb"
}

$result = Get-Content -Raw -Encoding UTF8 $resultPath | ConvertFrom-Json
$states = @($result.animation.state_sequence)
$lengths = @($result.onestring_stats.string_length)
$commands = @($result.onestring_stats.command_length)
$violations = @($result.onestring_stats.constraint_violation)
if ($states.Count -lt 2) { throw "ABD result contains too few animation states" }
if ($lengths.Count -ne $states.Count) {
    throw "OneString stats count does not match animation state count ($($lengths.Count) vs $($states.Count))"
}
if ($commands.Count -ne $states.Count -or $violations.Count -ne $states.Count) {
    throw "OneString command/violation logs are incomplete"
}
$finalLength = [double]$lengths[-1]
$finalViolation = [double]$violations[-1]

if ([double]::IsNaN($finalLength) -or [double]::IsInfinity($finalLength)) {
    throw "Final string length is not finite"
}
if ([double]::IsNaN($finalViolation) -or [double]::IsInfinity($finalViolation)) {
    throw "Final constraint violation is not finite"
}

$env:ONESTRING_ABD_EXECUTABLE = $exe
Write-Host ""
Write-Host "OneString ABD smoke verification passed." -ForegroundColor Green
Write-Host "States: $($states.Count)"
Write-Host "Initial length: $([double]$lengths[0])"
Write-Host "Final length:   $([double]$lengths[-1])"
Write-Host "Final command:  $([double]$commands[-1])"
Write-Host "Final violation:$([double]$violations[-1])"
Write-Host "Executable: $exe"
Write-Host "Current-shell ONESTRING_ABD_EXECUTABLE was set."
