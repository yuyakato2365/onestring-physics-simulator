param(
    [int]$Port = 8502,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

# scripts/run_split_panels.ps1 lives one directory below the Streamlit project.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python was not found. Run scripts\install_python_environment.ps1 first, or add Python to PATH."
    }
    $Python = $PythonCommand.Source
}

if ($Device -eq "auto") {
    $CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
    if ($LASTEXITCODE -eq 0 -and ($CudaAvailable | Select-Object -Last 1).Trim() -eq "1") {
        $Device = "cuda"
    } else {
        $Device = "cpu"
    }
}

$env:ONESTRING_BIJECTIVE_DEVICE = $Device

Write-Host "[OneString] project: $ProjectRoot"
Write-Host "[OneString] python:  $Python"
Write-Host "[OneString] device:  $env:ONESTRING_BIJECTIVE_DEVICE"
Write-Host "[OneString] app:     app_split_panels.py"
Write-Host "[OneString] url:     http://localhost:$Port"

& $Python -m streamlit run (Join-Path $ProjectRoot "app_split_panels.py") --server.port $Port
exit $LASTEXITCODE
