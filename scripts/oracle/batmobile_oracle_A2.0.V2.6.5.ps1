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
  Write-Host "🜁 TRUTH PROTOCOL INLINE STATE (COPY/PASTE READY)"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 18
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
    Get-ChildItem $r -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $nv = Join-Path $_.FullName "bin\nvcc.exe"
      if(Test-Path $nv){$found += $nv}
    }
  }
  return @($found | Select-Object -Unique)
}

try{

  # ───────────────────────────────────────────────────────────────────────────
  # ROOT ANCHOR
  # ───────────────────────────────────────────────────────────────────────────
  Set-Location $RepoDir
  Ensure-Dir "artifacts"

  $oraclePath  = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $reflectPath = Join-Path $RepoDir "artifacts\oracle_reflect.json"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — TRUTH PROTOCOL CANON BEGIN" -ForegroundColor Cyan

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
  # PRIMARY GATE + SMART FEEDBACK
  # ───────────────────────────────────────────────────────────────────────────
  $status="PASS"
  $smart=@()

  if($nvccDisk.Count -eq 0){
    $status="FAIL_CUDA_TOOLKIT_MISSING"
    $smart += "STOP: Torch CUDA runtime TRUE but nvcc.exe missing."
    $smart += "ACTION: Install NVIDIA CUDA Toolkit (Developer) with nvcc checked."
    $smart += "Reboot → rerun Oracle."
  }

  if($status -eq "PASS" -and -not $cudaPath){
    $status="FAIL_CUDA_PATH_MISSING"
    $smart += "CUDA Toolkit exists but CUDA_PATH env var missing."
    $smart += "Fix by reinstalling toolkit or setting CUDA_PATH manually."
  }

  # ───────────────────────────────────────────────────────────────────────────
  # GIT AUTOSAVE NODE (GUARDED)
  # ───────────────────────────────────────────────────────────────────────────
  $gitState="SKIPPED"
  if(Test-Path "$RepoDir\.git"){
    git add artifacts | Out-Null
    $changes = git diff --cached --name-only
    if($changes){
      if($changes.Trim() -ne ""){
        git commit -m "BATMOBILE Oracle $Tag — truth snapshot" | Out-Null
        git pull --rebase | Out-Null
        git push | Out-Null
        $gitState="SYNCED"
      }
    }
  }

  # ───────────────────────────────────────────────────────────────────────────
  # ROOTMIRROR VERIFIER NODE
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
  # ORACLE STATE OBJECT (CANON)
  # ───────────────────────────────────────────────────────────────────────────
  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")

    truth_protocol="CANON_LOCKED"

    status=$status
    SMART_FEEDBACK=@($smart)

    torch_cuda=$torchCuda

    nvcc=@{
      primary=$(if($nvccDisk.Count -gt 0){$nvccDisk[0]}else{$null})
      disk_candidates=@($nvccDisk)
    }

    cuda_env=@{
      CUDA_PATH=$cudaPath
    }

    git=@{
      state=$gitState
    }

    rootmirror=@{
      state=$rootmirror
      local_hash=$localHash
      remote_hash=$remoteHash
    }

    next_gate="Only if PASS → A2.0.V2.7.0 Extension Build Proof Node"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # EMIT + REFLECT OVERWRITE (MANDATORY)
  # ───────────────────────────────────────────────────────────────────────────
  $oracle | ConvertTo-Json -Depth 18 |
    Out-File -FilePath $oraclePath -Encoding UTF8 -Force

  $oracle | ConvertTo-Json -Depth 18 |
    Out-File -FilePath $reflectPath -Encoding UTF8 -Force

  Print-Inline $oracle

  Write-Host ""
  Write-Host "🜁 Oracle saved → $oraclePath" -ForegroundColor Green
  Write-Host "🜁 Reflect overwritten → $reflectPath" -ForegroundColor Yellow
  Write-Host "⧉ RootMirror → $rootmirror" -ForegroundColor Cyan
  Write-Host "✅ TRUTH PROTOCOL CANON COMPLETE" -ForegroundColor Green

}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
