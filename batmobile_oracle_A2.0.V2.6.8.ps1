param([string]$RepoDir,[string]$Tag)

$ErrorActionPreference="Stop"
$StartDir=(Get-Location).Path
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

function Ensure-Dir($p){
  if(-not(Test-Path $p)){
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

function Repo-IsDirty {
  $s = git status --porcelain
  if($s){ return $true }
  return $false
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

  Set-Location $RepoDir
  Ensure-Dir "artifacts"
  Ensure-Dir "docs\canon"

  $oraclePath   = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $reflectPath  = Join-Path $RepoDir "artifacts\oracle_reflect.json"
  $nvccProofTxt = Join-Path $RepoDir "artifacts\nvcc_version_$Tag.txt"
  $passToken    = Join-Path $RepoDir "artifacts\PASS_$Tag.token"
  $canonDoc     = Join-Path $RepoDir "docs\canon\TRUTH_PROTOCOL_CANON.md"

  $status="PASS"
  $smart=@()

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — TRUE EVOLUTION LOCK BEGIN" -ForegroundColor Cyan

  # ─────────────────────────────────────────────────────────────
  # DISK TRUTH
  # ─────────────────────────────────────────────────────────────
  $nvccDisk = Find-NVCC
  $cudaPath = $env:CUDA_PATH

  if($nvccDisk.Count -eq 0){
    $status="FAIL_CUDA_TOOLKIT_MISSING"
    $smart += "STOP: nvcc.exe missing."
    $smart += "ACTION: Install CUDA Toolkit (Developer) + reboot."
  }

  if($status -eq "PASS" -and -not $cudaPath){
    $status="FAIL_CUDA_PATH_MISSING"
    $smart += "STOP: CUDA_PATH env var missing."
  }

  # ─────────────────────────────────────────────────────────────
  # NVCC VERSION PROOF
  # ─────────────────────────────────────────────────────────────
  if($status -eq "PASS"){
    (& $nvccDisk[0] --version 2>&1) |
      Out-File $nvccProofTxt -Encoding UTF8 -Force
  }

  # ─────────────────────────────────────────────────────────────
  # CANON SHADOW HEADER LOCK
  # ─────────────────────────────────────────────────────────────
@"
# BATMOBILE.EVO — TRUTH PROTOCOL CANON (LOCKED)
Tag: $Tag
Timestamp: $(Get-Date -Format o)

LAW: Detect → Verify → Proof → Canon → Git → RootMirror → Token → Return
"@ | Out-File $canonDoc -Encoding UTF8 -Force

  # ─────────────────────────────────────────────────────────────
  # DIRTY TREE TOTAL BLOCK
  # ─────────────────────────────────────────────────────────────
  $gitState="SKIPPED"
  if(Test-Path "$RepoDir\.git"){
    if(Repo-IsDirty){
      $status="FAIL_DIRTY_TREE"
      $gitState="BLOCKED"
      $smart += "STOP: Repo dirty. No git actions allowed."
    }
  }

  # ─────────────────────────────────────────────────────────────
  # GIT ONLY IF PASS
  # ─────────────────────────────────────────────────────────────
  if($status -eq "PASS"){
    git add artifacts docs/canon
    $changes = git diff --cached --name-only
    if($changes){
      git commit -m "BATMOBILE Oracle $Tag — evolution lock"
      git push
      $gitState="SYNCED"
    }
  }

  # ─────────────────────────────────────────────────────────────
  # ROOTMIRROR HARD GATE
  # ─────────────────────────────────────────────────────────────
  $rootmirror="SKIPPED"
  if($status -eq "PASS"){
    $localHash=(git rev-parse HEAD).Trim()
    $remoteHash=(git ls-remote origin HEAD).Split("`t")[0].Trim()

    if($localHash -ne $remoteHash){
      $status="FAIL_ROOTMIRROR_DRIFT"
      $rootmirror="DRIFT"
      $smart += "STOP: RootMirror drift detected."
    } else {
      $rootmirror="VERIFIED"
    }
  }

  # ─────────────────────────────────────────────────────────────
  # PASS TOKEN LAST NODE (ONLY FINAL PASS)
  # ─────────────────────────────────────────────────────────────
  if($status -eq "PASS" -and $rootmirror -eq "VERIFIED"){
    "PASS:$Tag $(Get-Date -Format o)" |
      Out-File $passToken -Encoding UTF8 -Force
  }

  # ─────────────────────────────────────────────────────────────
  # ORACLE STATE EMIT
  # ─────────────────────────────────────────────────────────────
  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    status=$status
    SMART_FEEDBACK=@($smart)
    nvcc=@{ primary=$nvccDisk[0] }
    git=@{ state=$gitState }
    rootmirror=$rootmirror
    canon=@{ doc=$canonDoc; token=$passToken }
    next_gate="Only if PASS token exists → A2.0.V2.7.0"
  }

  $oracle | ConvertTo-Json -Depth 25 |
    Out-File $oraclePath -Encoding UTF8 -Force

  $oracle | ConvertTo-Json -Depth 25 |
    Out-File $reflectPath -Encoding UTF8 -Force

  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  $oracle | ConvertTo-Json -Depth 25
  Write-Host "══════════════════════════════════════════════════════════════"

  if($status -eq "PASS"){
    Write-Host "✅ $Tag COMPLETE (PASS)" -ForegroundColor Green
  } else {
    Write-Host "❌ $Tag COMPLETE (FAIL: $status)" -ForegroundColor Red
  }

}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
