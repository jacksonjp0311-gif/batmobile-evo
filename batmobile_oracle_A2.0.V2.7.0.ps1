param([string]$RepoDir,[string]$Tag)

$ErrorActionPreference="Stop"
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$StartDir=(Get-Location).Path

function Ensure-Dir($p){
  if(-not(Test-Path $p)){
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

function Print-Inline($Obj){
  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  Write-Host "🜁 ORACLE INLINE STATE (COPY/PASTE READY)"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 30
  Write-Host "══════════════════════════════════════════════════════════════"
}

function Find-NVCC {
  $roots=@(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
  )
  $found=@()
  foreach($r in $roots){
    if(-not(Test-Path $r)){continue}
    Get-ChildItem $r -Directory -ErrorAction SilentlyContinue | ForEach-Object{
      $nv=Join-Path $_.FullName "bin\nvcc.exe"
      if(Test-Path $nv){$found+=$nv}
    }
  }
  return @($found | Select-Object -Unique)
}

function Dirty-Files {
  try{
    $out = git status --porcelain
    if(-not $out){ return @() }
    return @($out)
  } catch { return @("GIT_STATUS_ERROR") }
}

try{
  Set-Location $RepoDir

  Ensure-Dir "artifacts"
  Ensure-Dir "docs\canon"

  $oraclePath   = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $reflectPath  = Join-Path $RepoDir "artifacts\oracle_reflect.json"
  $canonDoc     = Join-Path $RepoDir "docs\canon\TRUTH_PROTOCOL_CANON.md"
  $nextTxt      = Join-Path $RepoDir "artifacts\NEXT_ACTIONS_$Tag.txt"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — CLEANROOM PREP ORACLE BEGIN" -ForegroundColor Cyan

  # ─────────────────────────────────────────────────────────────
  # DETECT
  # ─────────────────────────────────────────────────────────────
  $nvccDisk = Find-NVCC
  $primaryNvcc = if($nvccDisk.Count -gt 0){$nvccDisk[0]}else{$null}

  $cudaPath = $env:CUDA_PATH
  $dirtyList = Dirty-Files

  $status="PASS_READY"
  $smart=@()

  # ─────────────────────────────────────────────────────────────
  # DIRTY TREE RESOLUTION NODE
  # ─────────────────────────────────────────────────────────────
  if($dirtyList.Count -gt 0){
    $status="BLOCKED_DIRTY_TREE"
    $smart += "STOP: Repo is dirty."
    $smart += "ACTION: Run → git add -A ; git commit -m `"WIP clean`""
    $smart += "Dirty files listed in oracle."
  }

  # ─────────────────────────────────────────────────────────────
  # CUDA INSTALL READINESS NODE
  # ─────────────────────────────────────────────────────────────
  if(-not $primaryNvcc){
    $status="FAIL_CUDA_TOOLKIT_MISSING"
    $smart += "STOP: nvcc.exe missing."
    $smart += "ACTION:"
    $smart += "1) Install NVIDIA CUDA Toolkit (Developer)"
    $smart += "2) Ensure CUDA Compiler (nvcc) checked"
    $smart += "3) Reboot"
    $smart += "4) Rerun Oracle"
  }

  if($status -eq "PASS_READY" -and -not $cudaPath){
    $status="FAIL_CUDA_PATH_MISSING"
    $smart += "STOP: CUDA Toolkit found but CUDA_PATH missing."
    $smart += "Fix: reinstall toolkit or set CUDA_PATH manually."
  }

  # ─────────────────────────────────────────────────────────────
  # WRITE NEXT ACTIONS TXT (MANDATORY)
  # ─────────────────────────────────────────────────────────────
  $smart | Out-File $nextTxt -Encoding UTF8 -Force

  # ─────────────────────────────────────────────────────────────
  # CANON LOCK UPDATE
  # ─────────────────────────────────────────────────────────────
@"
# BATMOBILE.EVO — CANON LOCK (A2.0.V2.7.0)

This node is CLEANROOM PREP.

- No extension builds allowed yet.
- Goal: unblock repo + install nvcc.
- Next evolution forbidden until PASS token exists.

LAW: Detect → Diagnose → SmartFeedback → Reflect → Canon → Return
"@ | Out-File $canonDoc -Encoding UTF8 -Force

  # ─────────────────────────────────────────────────────────────
  # ORACLE STATE OBJECT
  # ─────────────────────────────────────────────────────────────
  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    truth_protocol="CLEANROOM_PREP_v2.7.0"

    status=$status
    SMART_FEEDBACK=@($smart)

    dirty_tree=@($dirtyList)

    nvcc=@{
      primary=$primaryNvcc
      disk_candidates=@($nvccDisk)
    }

    cuda_env=@{
      CUDA_PATH=$cudaPath
    }

    artifacts=@{
      next_actions=$nextTxt
      canon_doc=$canonDoc
    }

    next_gate="Only if CLEAN + nvcc exists → A2.0.V2.7.1 PASS TOKEN ORACLE"
  }

  $oracle | ConvertTo-Json -Depth 30 | Out-File $oraclePath -Encoding UTF8 -Force
  $oracle | ConvertTo-Json -Depth 30 | Out-File $reflectPath -Encoding UTF8 -Force

  Print-Inline $oracle

  Write-Host ""
  Write-Host "🜁 Oracle saved → $oraclePath" -ForegroundColor Green
  Write-Host "🜁 Next actions → $nextTxt" -ForegroundColor Yellow
  Write-Host "✅ $Tag COMPLETE (status=$status)" -ForegroundColor Green

}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
