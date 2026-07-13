param(
    [string]$PythonVersion = "3.11",
    [switch]$WithDevDependencies
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Windows Python launcher 'py' was not found. Install Python 3.11+ first."
}

if (-not (Test-Path -LiteralPath ".venv/Scripts/python.exe")) {
    & py "-$PythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv with Python $PythonVersion"
    }
}

$python = [System.IO.Path]::GetFullPath(".venv/Scripts/python.exe")
& $python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

$target = if ($WithDevDependencies) { ".[dev]" } else { "." }
& $python -m pip install -e $target
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed" }

Write-Host "Python environment ready: $python" -ForegroundColor Green
Write-Host "Run: .\.venv\Scripts\python.exe -m streamlit run app.py"
