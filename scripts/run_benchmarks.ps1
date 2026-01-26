# BATMOBILE.EVO — Benchmarks Runner (A2.0.V1.3)
# Artifact truth only. No claims.

param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Split-Path -Parent (Split-Path -Parent $($MyInvocation.MyCommand.Path)))
Set-Location -LiteralPath ""

Write-Host ""
Write-Host "𓏤 [BATMOBILE] Phase 3 benchmark harness → A2.0.V1.3"
Write-Host ""

# Optional: minimal dependency check (do not force install; record failures in JSON)
& $Python -c "import sys; print(sys.version)" | Out-Host

$env:BATMOBILE_TAG = "A2.0.V1.3"

$h = Join-Path $RepoRoot "benchmarks\harness\run_harness.py"
if (-not (Test-Path -LiteralPath $h)) {
  throw "Harness missing: $h"
}

& $Python $h
if ($LASTEXITCODE -ne 0) {
  Write-Host "❌ Harness failed (see benchmarks/results stderr artifacts)" -ForegroundColor Red
  exit 1
}

Write-Host "🜁 Benchmarks complete (results written under benchmarks/results)." -ForegroundColor Green
