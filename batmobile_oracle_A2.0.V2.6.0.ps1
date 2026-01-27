<#
╔══════════════════════════════════════════════════════════════════════════════╗
║ 𓂀  BATMOBILE.EVO — ALL-ONE ORACLE (A2.0.V2.6.0)                              ║
║                                                                              ║
║ CUDA EXTENSION PROOF ORACLE — NVCC GATE ACTIVATION                           ║
║                                                                              ║
║ REQUIRED BACKEND VERIFIERS                                                   ║
║  • NVCC_VERIFIER        — nvcc.exe truth                                     ║
║  • CUDA_HOME_VERIFIER   — CUDA_PATH truth                                    ║
║  • TORCH_CUDA_VERIFIER  — torch.cuda.is_available() truth                    ║
║                                                                              ║
║ NEXT GATE                                                                    ║
║  If PASS → A2.0.V2.7.0 Extension Build Proof Node                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
#>

param(
  [string]$RepoDir,
  [string]$Tag
)

$ErrorActionPreference = "Stop"
$StartDir = (Get-Location).Path
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Helpers (Dirs + Inline JSON)
# ─────────────────────────────────────────────────────────────────────────────
function Ensure-Dir($p) {
  if (-not (Test-Path -LiteralPath $p)) {
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

function Print-Inline($Obj) {
  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  Write-Host "🜁 CUDA ORACLE INLINE STATE (COPY/PASTE READY)"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 10
  Write-Host "══════════════════════════════════════════════════════════════"
}

function Tool-Path($Name) {
  $c = Get-Command $Name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  return $null
}

try {

  Set-Location $RepoDir
  Ensure-Dir "artifacts"

  $oraclePath = Join-Path $RepoDir "artifacts\oracle_$Tag.json"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — CUDA ORACLE BEGIN" -ForegroundColor Cyan

  # ───────────────────────────────────────────────────────────────────────────
  # VERIFIER NODE 1 — NVCC_VERIFIER
  # ───────────────────────────────────────────────────────────────────────────
  $nvccPath = Tool-Path "nvcc.exe"

  # ───────────────────────────────────────────────────────────────────────────
  # VERIFIER NODE 2 — CUDA_HOME_VERIFIER
  # ───────────────────────────────────────────────────────────────────────────
  $cudaHome = $env:CUDA_PATH

  # ───────────────────────────────────────────────────────────────────────────
  # VERIFIER NODE 3 — TORCH_CUDA_VERIFIER
  # ───────────────────────────────────────────────────────────────────────────
  $torchCUDA = "UNKNOWN"
  try {
    $torchCUDA = python -c "import torch; print(torch.cuda.is_available())"
  } catch {
    $torchCUDA = "PYTHON_OR_TORCH_MISSING"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 1 — Gate Classification
  # ───────────────────────────────────────────────────────────────────────────
  $status   = "OK"
  $guidance = @()

  if (-not $nvccPath) {
    $status = "NVCC_MISSING"
    $guidance += "Install NVIDIA CUDA Toolkit (nvcc.exe not found)"
  }

  if ($status -eq "OK" -and -not $cudaHome) {
    $status = "CUDA_HOME_MISSING"
    $guidance += "CUDA_PATH not set — reinstall CUDA Toolkit properly"
  }

  if ($status -eq "OK" -and $torchCUDA -notmatch "True") {
    $status = "TORCH_CUDA_NOT_READY"
    $guidance += "Torch does not detect CUDA"
    $guidance += "Install torch with CUDA build (pip/conda correct version)"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 2 — Oracle State Object
  # ───────────────────────────────────────────────────────────────────────────
  $oracle = @{
    tag   = $Tag
    time  = (Get-Date).ToString("o")

    status = $status

    nvcc = @{
      path = $nvccPath
    }

    cuda_home = $cudaHome

    torch = @{
      cuda_available = $torchCUDA
    }

    guidance = @($guidance)

    next_gate = "If PASS → A2.0.V2.7.0 Extension Build Proof Node"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 3 — Emit Artifact + Print Inline
  # ───────────────────────────────────────────────────────────────────────────
  $oracle | ConvertTo-Json -Depth 10 |
    Out-File -FilePath $oraclePath -Encoding UTF8 -Force

  Print-Inline $oracle

  Write-Host ""
  Write-Host "🜁 Oracle saved → $oraclePath" -ForegroundColor Green
  Write-Host "✅ CUDA ORACLE COMPLETE" -ForegroundColor Green

}
finally {
  Set-Location $StartDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
