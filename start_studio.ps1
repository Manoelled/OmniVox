$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at $Python"
    exit 1
}

Write-Host "Starting OmniVox at http://127.0.0.1:7860 ..." -ForegroundColor Cyan

Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:7860"
} | Out-Null

& $Python -m uvicorn app.server:app --host 127.0.0.1 --port 7860 --reload --reload-dir app
