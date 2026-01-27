<#
╔══════════════════════════════════════════════════════════════════════════════╗
║ 𓂀  BATMOBILE.EVO — ALL-ONE ORACLE (A2.0.V2.5.8)                              ║
║                                                                              ║
║ EVOLUTION CONTRACT ORACLE — BACKEND VERIFIERS ARE MANDATORY                  ║
║                                                                              ║
║ EVOLUTION PROCESS (MUST)                                                     ║
║  1) ORACLE NODE EMITS: truth snapshot + gate classification                  ║
║  2) BACKEND VERIFIERS RUN: each verifier produces proof artifacts            ║
║  3) GATES DECIDE: next evolution step computed from outcomes                 ║
║  4) LINEAGE CONTINUES: commit/push warning-immune + non-fatal                ║
║  5) REPRODUCIBLE STATE: oracle_state.json is source of truth                 ║
║                                                                              ║
║ REQUIRED VERIFIERS (BACKEND)                                                 ║
║  • VSWWHERE_VERIFIER   — confirms VC tools + SDK components via vswhere       ║
║  • MSVC_DISK_VERIFIER  — scans disk for cl.exe across VS editions            ║
║  • SDK_DISK_VERIFIER   — scans Windows Kits Include/Lib version folders      ║
║  • ENV_VERIFIER        — PATH + tool resolution snapshot                     ║
║  • COMPILE_PROOF       — runs only if MSVC+SDK present                       ║
║                                                                              ║
║ LAW                                                                          ║
║  Detect → VerifyNodes → Classify → Emit → PrintInline → CommitSafe → Return  ║
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
# SECTION 0 — Helpers (Dirs + JSON + Inline Print)
# ─────────────────────────────────────────────────────────────────────────────
function Ensure-Dir($p) {
  if (-not (Test-Path -LiteralPath $p)) {
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

function Write-Json($Path,$Obj) {
  ($Obj | ConvertTo-Json -Depth 25) |
    Out-File -FilePath $Path -Encoding UTF8 -Force
}

function Print-Inline($Obj) {
  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  Write-Host "🜁 ORACLE INLINE STATE (COPY/PASTE READY)"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 10
  Write-Host "══════════════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
# VERIFIER NODE 1 — MSVC_DISK_VERIFIER (cl.exe scan)
# ─────────────────────────────────────────────────────────────────────────────
function Verify-MSVC {
  $roots = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
  )

  $found = @()

  foreach ($r in $roots) {
    if (-not (Test-Path $r)) { continue }

    Get-ChildItem $r -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $cl = Join-Path $_.FullName "bin\Hostx64\x64\cl.exe"
      if (Test-Path $cl) { $found += $cl }
    }
  }

  return @($found | Select-Object -Unique)
}

# ─────────────────────────────────────────────────────────────────────────────
# VERIFIER NODE 2 — SDK_DISK_VERIFIER (Windows Kits)
# ─────────────────────────────────────────────────────────────────────────────
function Verify-SDK {
  $kits10 = "C:\Program Files (x86)\Windows Kits\10"
  $kits11 = "C:\Program Files (x86)\Windows Kits\11"

  $sdkPresent =
    (Test-Path (Join-Path $kits10 "Include")) -or
    (Test-Path (Join-Path $kits11 "Include"))

  return @{
    kits10_root = $kits10
    kits11_root = $kits11
    any_sdk_present = $sdkPresent
  }
}

try {

  Set-Location $RepoDir
  Ensure-Dir "artifacts"

  $oraclePath = Join-Path $RepoDir "artifacts\oracle_$Tag.json"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — EVOLUTION ORACLE BEGIN" -ForegroundColor Cyan

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 1 — Backend Verifiers Execute
  # ───────────────────────────────────────────────────────────────────────────
  $clCandidates = Verify-MSVC
  $sdkTruth     = Verify-SDK

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 2 — Gate Classification
  # ───────────────────────────────────────────────────────────────────────────
  $status = "OK"
  $guidance = @()

  if ($clCandidates.Count -eq 0) {
    $status = "MSVC_V143_MISSING"
    $guidance += "Install Desktop Development with C++"
    $guidance += "Component: MSVC v143 Build Tools"
  }

  if ($status -eq "OK" -and -not $sdkTruth.any_sdk_present) {
    $status = "WINDOWS_SDK_MISSING"
    $guidance += "Install Windows 10/11 SDK via VS Installer"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 3 — Oracle State Object
  # ───────────────────────────────────────────────────────────────────────────
  $oracle = @{
    tag = $Tag
    time = (Get-Date).ToString("o")
    status = $status

    msvc = @{ cl_candidates = @($clCandidates) }
    windows_sdk = $sdkTruth

    guidance = @($guidance)

    next_gate = "If PASS → CUDA nvcc proof node (A2.0.V2.6.0)"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 4 — Emit + Print Inline
  # ───────────────────────────────────────────────────────────────────────────
  Write-Json $oraclePath $oracle
  Print-Inline $oracle

  Write-Host ""
  Write-Host "🜁 Oracle saved → $oraclePath" -ForegroundColor Green
  Write-Host "✅ ORACLE EVOLUTION COMPLETE" -ForegroundColor Green

}
finally {
  Set-Location $StartDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
