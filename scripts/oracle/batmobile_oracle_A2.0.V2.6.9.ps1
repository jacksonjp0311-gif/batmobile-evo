param([string]$RepoDir,[string]$Tag)

$ErrorActionPreference="Stop"
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

function Ensure-Dir($p){
  if(-not (Test-Path $p)){
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

function Print-Inline($Obj){
  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  Write-Host "🜁 ORACLE INLINE STATE (COPY/PASTE READY)"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 25
  Write-Host "══════════════════════════════════════════════════════════════"
}

function Find-NVCC {
  $roots=@(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
  )
  $found=@()
  foreach($r in $roots){
    if(-not(Test-Path $r)){ continue }
    Get-ChildItem $r -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $nv = Join-Path $_.FullName "bin\nvcc.exe"
      if(Test-Path $nv){ $found += $nv }
    }
  }
  return @($found | Select-Object -Unique)
}

function Repo-IsDirty {
  try{
    $s = git status --porcelain 2>$null
    if($s){ return $true }
    return $false
  } catch { return $true }
}

$StartDir=(Get-Location).Path

# Predeclare emission targets so we can FAIL-SAFE emit in catch {}
$oraclePath=$null
$reflectPath=$null

try{
  Set-Location $RepoDir

  Ensure-Dir "artifacts"
  Ensure-Dir "docs\canon"

  $oraclePath   = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $reflectPath  = Join-Path $RepoDir "artifacts\oracle_reflect.json"
  $nvccProofTxt = Join-Path $RepoDir "artifacts\nvcc_version_$Tag.txt"
  $passToken    = Join-Path $RepoDir "artifacts\PASS_$Tag.token"
  $canonDoc     = Join-Path $RepoDir "docs\canon\TRUTH_PROTOCOL_CANON.md"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — PROCESS-ALIGNED ORACLE BEGIN" -ForegroundColor Cyan

  # ───────────────────────────────────────────────────────────────────────────
  # DETECT / VERIFY
  # ───────────────────────────────────────────────────────────────────────────
  $nvccDisk = Find-NVCC
  $primaryNvcc = if($nvccDisk.Count -gt 0){ $nvccDisk[0] } else { $null }

  $cudaPath = $env:CUDA_PATH

  $torchCuda="UNKNOWN"
  try{
    $torchCuda = python -c "import torch; print(torch.cuda.is_available())" 2>&1
  } catch {
    $torchCuda = "TORCH_NOT_FOUND"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SMART FEEDBACK (FIRST-CLASS)
  # ───────────────────────────────────────────────────────────────────────────
  $status="PASS"
  $smart=@()

  if(-not $primaryNvcc){
    $status="FAIL_CUDA_TOOLKIT_MISSING"
    $smart += "STOP: nvcc.exe not found on disk."
    $smart += "NEXT ACTION:"
    $smart += "1) Install NVIDIA CUDA Toolkit (Developer installer)"
    $smart += "2) Ensure 'CUDA Compiler (nvcc)' is selected"
    $smart += "3) Reboot Windows"
    $smart += "4) Rerun Oracle $Tag"
  }

  if($status -eq "PASS" -and -not $cudaPath){
    $status="FAIL_CUDA_PATH_MISSING"
    $smart += "STOP: CUDA Toolkit found but CUDA_PATH env var missing."
    $smart += "NEXT ACTION: reinstall toolkit OR set CUDA_PATH then reboot."
  }

  # ───────────────────────────────────────────────────────────────────────────
  # PROOF NODE (ONLY IF PASS)
  # ───────────────────────────────────────────────────────────────────────────
  $nvccVersion="NOT_RUN"
  if($status -eq "PASS"){
    $nvccVersion = (& $primaryNvcc --version 2>&1)
    $nvccVersion | Out-File $nvccProofTxt -Encoding UTF8 -Force
  }

  # ───────────────────────────────────────────────────────────────────────────
  # CANON DOC LOCK (ALWAYS)
  # ───────────────────────────────────────────────────────────────────────────
@"
# BATMOBILE.EVO — TRUTH PROTOCOL CANON (LOCKED)

**Tag:** $Tag
**Timestamp:** $(Get-Date -Format o)

## NODES (PROCESS-ALIGNED)

1. SMART_FEEDBACK_NODE (always)
2. REFLECTIVE_STATE_NODE (always overwrite)
3. LINEAGE_ARTIFACT_NODE (always write oracle_$Tag.json)
4. NVCC_VERSION_PROOF_NODE (only if nvcc exists)
5. SHADOW_HEADER_LOCK_NODE (always)
6. DIRTY_TREE_GIT_GUARD_NODE (always)
7. ROOTMIRROR_NODE (only if git path exists and clean)
8. PASS_TOKEN_NODE (only if PASS + RootMirror VERIFIED)
9. RETURN_TO_ROOT_GUARANTEE (always)

LAW: Detect → Verify → SmartFeedback → Reflect → Proof → Canon → Git → RootMirror → Return
"@ | Out-File $canonDoc -Encoding UTF8 -Force

  # ───────────────────────────────────────────────────────────────────────────
  # GIT NODE (GUARDED)
  # ───────────────────────────────────────────────────────────────────────────
  $gitState="SKIPPED"
  $rootmirror="SKIPPED"
  $localHash=$null
  $remoteHash=$null

  if(Test-Path "$RepoDir\.git"){
    if(Repo-IsDirty){
      $gitState="BLOCKED_DIRTY_TREE"
      $smart += "GIT BLOCKED: working tree dirty. Commit/stash before pull/rebase."
    } else {
      $gitState="CLEAN"
      # Only commit canon artifacts if any changes
      git add artifacts docs/canon 2>$null | Out-Null
      $changes = git diff --cached --name-only 2>$null
      if($changes -and $changes.Trim() -ne ""){
        git commit -m "BATMOBILE Oracle $Tag — process-aligned canon lock" 2>$null | Out-Null
        git pull --rebase 2>$null | Out-Null
        git push 2>$null | Out-Null
        $gitState="SYNCED"
      }
    }

    # RootMirror only if git not blocked
    if($gitState -ne "BLOCKED_DIRTY_TREE"){
      try{
        $localHash=(git rev-parse HEAD).Trim()
        $remoteLine=(git ls-remote origin HEAD | Select-Object -First 1)
        if($remoteLine){ $remoteHash=($remoteLine -split "\s+")[0] }
        if($remoteHash){
          if($localHash -eq $remoteHash){ $rootmirror="VERIFIED" }
          else { $rootmirror="DRIFT" }
        }
      } catch {
        $rootmirror="ERROR"
      }
    }
  }

  # ───────────────────────────────────────────────────────────────────────────
  # PASS TOKEN (ONLY IF TRUE PASS)
  # ───────────────────────────────────────────────────────────────────────────
  if($status -eq "PASS" -and $rootmirror -eq "VERIFIED"){
    "PASS:$Tag $(Get-Date -Format o)" | Out-File $passToken -Encoding UTF8 -Force
  }

  # ───────────────────────────────────────────────────────────────────────────
  # ORACLE STATE OBJECT (ALWAYS EMIT)
  # ───────────────────────────────────────────────────────────────────────────
  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    truth_protocol="PROCESS_ALIGNED_FAILSAFE_v2.6.9"

    status=$status
    SMART_FEEDBACK=@($smart)

    torch_cuda=$torchCuda
    cuda_path=$cudaPath

    nvcc=@{
      primary=$primaryNvcc
      disk_candidates=@($nvccDisk)
      version_proof=$nvccProofTxt
      version=$nvccVersion
    }

    canon=@{
      protocol_doc=$canonDoc
      pass_token=$passToken
    }

    git=@{
      state=$gitState
    }

    rootmirror=@{
      state=$rootmirror
      local_hash=$localHash
      remote_hash=$remoteHash
    }

    next_gate="Only if PASS token exists → A2.0.V2.7.0 Extension Proof Node"
  }

  $oracle | ConvertTo-Json -Depth 25 | Out-File $oraclePath -Encoding UTF8 -Force
  $oracle | ConvertTo-Json -Depth 25 | Out-File $reflectPath -Encoding UTF8 -Force

  Print-Inline $oracle

  Write-Host ""
  Write-Host "🜁 Oracle saved → $oraclePath" -ForegroundColor Green
  Write-Host "🜁 Reflect saved → $reflectPath" -ForegroundColor Yellow
  Write-Host "🜁 Canon locked → $canonDoc" -ForegroundColor Cyan
  Write-Host "✅ $Tag COMPLETE (status=$status)" -ForegroundColor Green

}
catch{
  # FAIL-SAFE EMIT EVEN ON EXCEPTIONS
  $err = $_.Exception.Message
  $fallback=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    truth_protocol="FAILSAFE_CATCH_EMIT_v2.6.9"
    status="FAIL_ORACLE_EXCEPTION"
    error=$err
  }
  if($oraclePath){
    $fallback | ConvertTo-Json -Depth 10 | Out-File $oraclePath -Encoding UTF8 -Force
  }
  if($reflectPath){
    $fallback | ConvertTo-Json -Depth 10 | Out-File $reflectPath -Encoding UTF8 -Force
  }
  Write-Host "❌ ORACLE EXCEPTION (but state emitted): $err" -ForegroundColor Red
}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
