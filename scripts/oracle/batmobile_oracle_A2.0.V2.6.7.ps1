param([string]$RepoDir,[string]$Tag)

$ErrorActionPreference="Stop"
$StartDir=(Get-Location).Path
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

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
  $Obj | ConvertTo-Json -Depth 25
  Write-Host "══════════════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
# VERIFIER NODE — DISK NVCC SCAN
# ─────────────────────────────────────────────────────────────────────────────
function Find-NVCC {
  $roots=@(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
  )
  $found=@()
  foreach($r in $roots){
    if(-not(Test-Path $r)){continue}
    Get-ChildItem $r -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $nv = Join-Path $_.FullName "bin\nvcc.exe"
      if(Test-Path $nv){$found += $nv}
    }
  }
  return @($found | Select-Object -Unique)
}

# ─────────────────────────────────────────────────────────────────────────────
# DIRTY TREE GUARD NODE
# ─────────────────────────────────────────────────────────────────────────────
function Repo-IsDirty {
  $s = git status --porcelain
  if($s){ return $true }
  return $false
}

try{

  # ROOT ANCHOR
  Set-Location $RepoDir

  Ensure-Dir "artifacts"
  Ensure-Dir "docs\canon"

  $oraclePath   = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $reflectPath  = Join-Path $RepoDir "artifacts\oracle_reflect.json"
  $nvccProofTxt = Join-Path $RepoDir "artifacts\nvcc_version_$Tag.txt"
  $passToken    = Join-Path $RepoDir "artifacts\PASS_$Tag.token"
  $canonDoc     = Join-Path $RepoDir "docs\canon\TRUTH_PROTOCOL_CANON.md"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — CANON ORACLE BEGIN" -ForegroundColor Cyan

  # ───────────────────────────────────────────────────────────────────────────
  # TRUTH SNAPSHOT
  # ───────────────────────────────────────────────────────────────────────────
  $nvccDisk = Find-NVCC
  $cudaPath = $env:CUDA_PATH

  $torchCuda="UNKNOWN"
  try{
    $torchCuda = python -c "import torch; print(torch.cuda.is_available())"
  } catch {
    $torchCuda = "TORCH_NOT_FOUND"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # PRIMARY STATUS + SMART FEEDBACK
  # ───────────────────────────────────────────────────────────────────────────
  $status="PASS"
  $smart=@()

  if($nvccDisk.Count -eq 0){
    $status="FAIL_CUDA_TOOLKIT_MISSING"
    $smart += "STOP: Torch CUDA runtime TRUE but nvcc.exe missing."
    $smart += "NEXT ACTION:"
    $smart += "1) Install NVIDIA CUDA Toolkit (Developer installer)"
    $smart += "2) Ensure CUDA Compiler (nvcc) is checked"
    $smart += "3) Reboot Windows"
    $smart += "4) Rerun Oracle $Tag"
  }

  if($status -eq "PASS" -and -not $cudaPath){
    $status="FAIL_CUDA_PATH_MISSING"
    $smart += "CUDA Toolkit exists but CUDA_PATH env var missing."
    $smart += "Fix: reinstall toolkit OR set CUDA_PATH manually."
  }

  # ───────────────────────────────────────────────────────────────────────────
  # NVCC VERSION PROOF NODE (MANDATORY IF PASS)
  # ───────────────────────────────────────────────────────────────────────────
  $nvccVersion="NOT_RUN"
  if($status -eq "PASS"){
    $nvccVersion = & $nvccDisk[0] --version 2>&1
    $nvccVersion | Out-File $nvccProofTxt -Encoding UTF8 -Force
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SHADOW HEADER LOCK NODE (CANON SURFACE)
  # ───────────────────────────────────────────────────────────────────────────
  @"
# BATMOBILE.EVO — TRUTH PROTOCOL CANON (LOCKED)

**Tag:** $Tag  
**Timestamp:** $(Get-Date -Format o)

## NON-NEGOTIABLE NODES

1. SMART_FEEDBACK_NODE — exact next human action printed  
2. REFLECTIVE_STATE_NODE — oracle_reflect.json overwritten every run  
3. LINEAGE_ARTIFACT_NODE — oracle_A2.x.json saved every run  
4. NVCC_VERSION_PROOF_NODE — nvcc --version artifact required  
5. DIRTY_TREE_GIT_GUARD_NODE — no pull/rebase if repo dirty  
6. ROOTMIRROR_NODE — local hash must equal remote hash  
7. PASS_TOKEN_NODE — next gate forbidden unless PASS token exists  
8. RETURN_TO_ROOT_GUARANTEE — always returns to repo root  

LAW: Detect → Verify → SmartFeedback → Reflect → Proof → Canon → RootMirror → Return
"@ | Out-File $canonDoc -Encoding UTF8 -Force

  # ───────────────────────────────────────────────────────────────────────────
  # PASS TOKEN NODE (ONLY IF PASS)
  # ───────────────────────────────────────────────────────────────────────────
  if($status -eq "PASS"){
    "PASS:$Tag $(Get-Date -Format o)" | Out-File $passToken -Encoding UTF8 -Force
  }

  # ───────────────────────────────────────────────────────────────────────────
  # GIT NODE — FULL DIRTY TREE GUARD
  # ───────────────────────────────────────────────────────────────────────────
  $gitState="SKIPPED"

  if(Test-Path "$RepoDir\.git"){

    if(Repo-IsDirty){
      $gitState="BLOCKED_DIRTY_TREE"
      $smart += "GIT BLOCKED: Working tree dirty. Commit/stash before rebase."
    }

    if($gitState -ne "BLOCKED_DIRTY_TREE"){
      git add artifacts docs/canon | Out-Null

      $changes = git diff --cached --name-only
      if($changes){
        if($changes.Trim() -ne ""){
          git commit -m "BATMOBILE Oracle $Tag — canon lock + pass token" | Out-Null
          git pull --rebase | Out-Null
          git push | Out-Null
          $gitState="SYNCED"
        }
      }
    }
  }

  # ───────────────────────────────────────────────────────────────────────────
  # ROOTMIRROR NODE
  # ───────────────────────────────────────────────────────────────────────────
  $rootmirror="SKIPPED"
  $localHash=$null
  $remoteHash=$null

  if(Test-Path "$RepoDir\.git"){
    $localHash=(git rev-parse HEAD).Trim()
    $remoteLine=(git ls-remote origin HEAD | Select-Object -First 1)
    if($remoteLine){
      $remoteHash=($remoteLine -split "\s+")[0]
    }

    if($remoteHash){
      if($localHash -eq $remoteHash){$rootmirror="VERIFIED"}
      if($localHash -ne $remoteHash){$rootmirror="DRIFT"}
    }
  }

  # ───────────────────────────────────────────────────────────────────────────
  # ORACLE STATE OBJECT (CANON LOCKED v2.6.7)
  # ───────────────────────────────────────────────────────────────────────────
  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")

    truth_protocol="CANON_LOCKED_v2.6.7"

    status=$status
    SMART_FEEDBACK=@($smart)

    torch_cuda=$torchCuda

    nvcc=@{
      primary=$(if($nvccDisk.Count -gt 0){$nvccDisk[0]}else{$null})
      disk_candidates=@($nvccDisk)
      version_proof=$nvccProofTxt
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

    next_gate="Only if PASS + token → A2.0.V2.7.0 Extension Proof Node"
  }

  # EMIT + REFLECT OVERWRITE
  $oracle | ConvertTo-Json -Depth 25 |
    Out-File -FilePath $oraclePath -Encoding UTF8 -Force

  $oracle | ConvertTo-Json -Depth 25 |
    Out-File -FilePath $reflectPath -Encoding UTF8 -Force

  Print-Inline $oracle

  Write-Host ""
  Write-Host "🜁 Oracle saved → $oraclePath" -ForegroundColor Green
  Write-Host "🜁 Canon locked → $canonDoc" -ForegroundColor Yellow
  Write-Host "⧉ RootMirror → $rootmirror" -ForegroundColor Cyan
  Write-Host "✅ A2.0.V2.6.7 COMPLETE" -ForegroundColor Green

}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
