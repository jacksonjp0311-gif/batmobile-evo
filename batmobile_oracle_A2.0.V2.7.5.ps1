param([string]$RepoDir,[string]$Tag)

$ErrorActionPreference="Stop"
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

function Ensure-Dir($p){
  if(-not (Test-Path $p)){
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

function Safe-Write($Path,$Content){
  if([string]::IsNullOrWhiteSpace($Path)){
    throw "EMPTY_PATH_GUARD: attempted write with empty path."
  }
  $Content | Out-File $Path -Encoding UTF8 -Force
}

function Repo-IsDirty {
  try{
    $s = git status --porcelain
    if($s){ return $true }
    return $false
  } catch { return $true }
}

function Find-NVCC-Highest {
  $root="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
  if(-not (Test-Path $root)){ return $null }

  $candidates=@()
  Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | ForEach-Object{
    $nv=Join-Path $_.FullName "bin\nvcc.exe"
    if(Test-Path $nv){ $candidates += $_.FullName }
  }

  if($candidates.Count -eq 0){ return $null }

  $best = $candidates | Sort-Object -Descending | Select-Object -First 1
  return (Join-Path $best "bin\nvcc.exe")
}

function RootMirror-Verify {
  $localHash=$null
  $remoteHash=$null
  $state="SKIPPED"
  try{
    $localHash=(git rev-parse HEAD).Trim()
    $remoteLine=(git ls-remote origin HEAD | Select-Object -First 1)
    if($remoteLine){ $remoteHash=($remoteLine -split "\s+")[0] }
    if($remoteHash){
      if($localHash -eq $remoteHash){ $state="VERIFIED" } else { $state="DRIFT" }
    } else {
      $state="NO_REMOTE"
    }
  } catch {
    $state="ERROR"
  }
  return @{
    state=$state
    local_hash=$localHash
    remote_hash=$remoteHash
  }
}

function Print-Inline($Obj){
  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  Write-Host "🜁 POST-REBOOT COMPILE PROOF INLINE STATE"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 40
  Write-Host "══════════════════════════════════════════════════════════════"
}

$StartDir=(Get-Location).Path
$oraclePath=$null

try{
  Set-Location $RepoDir

  Ensure-Dir "artifacts"
  Ensure-Dir "build\proof_$Tag"

  $oraclePath        = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $reflectPath       = Join-Path $RepoDir "artifacts\oracle_reflect.json"
  $proofLog          = Join-Path $RepoDir "artifacts\nvcc_proof_$Tag.txt"
  $nvccVersionProof  = Join-Path $RepoDir "artifacts\nvcc_version_$Tag.txt"

  # Reboot tokens
  $rebootRequired    = Join-Path $RepoDir "artifacts\REBOOT_REQUIRED.token"
  $rebootConfirmed   = Join-Path $RepoDir "artifacts\REBOOT_CONFIRMED.token"

  # Pass token
  $passToken         = Join-Path $RepoDir "artifacts\PASS_$Tag.token"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — POST-REBOOT COMPILE PROOF BEGIN" -ForegroundColor Cyan

  $smart=@()
  $status="PASS_READY"

  # ─────────────────────────────────────────────────────────────
  # (1) REBOOT CONFIRM NODE
  # ─────────────────────────────────────────────────────────────
  if(-not (Test-Path $rebootRequired)){
    $status="FAIL_REBOOT_REQUIRED_TOKEN_MISSING"
    $smart += "STOP: REBOOT_REQUIRED token not found."
    $smart += "Expected → $rebootRequired"
    $smart += "Action: run A2.0.V2.7.4 repair node (it mints this token), then reboot."
  } else {
    # Mint REBOOT_CONFIRMED every post-reboot run
    Safe-Write $rebootConfirmed ("REBOOT_CONFIRMED:$Tag " + (Get-Date -Format o))
    $smart += "✅ REBOOT CONFIRMED TOKEN → $rebootConfirmed"
  }

  # ─────────────────────────────────────────────────────────────
  # (2) CLEAN TREE REQUIRED
  # ─────────────────────────────────────────────────────────────
  if($status -eq "PASS_READY" -and (Repo-IsDirty)){
    $status="BLOCKED_DIRTY_TREE"
    $smart += "STOP: Repo dirty. Commit/stash before compile proof."
  }

  # ─────────────────────────────────────────────────────────────
  # (3) NVCC HIGHEST VERSION NODE + VERSION PROOF
  # ─────────────────────────────────────────────────────────────
  $nvcc=$null
  $nvccVersion="NOT_RUN"
  if($status -eq "PASS_READY"){
    $nvcc = Find-NVCC-Highest
    if(-not $nvcc){
      $status="FAIL_NVCC_MISSING"
      $smart += "STOP: nvcc.exe missing even after reboot."
      $smart += "Action: reinstall CUDA Toolkit (Developer) with CUDA Compiler (nvcc)."
    } else {
      $nvccVersion = & $nvcc --version 2>&1
      Safe-Write $nvccVersionProof $nvccVersion
      $smart += "✅ NVCC VERSION PROOF → $nvccVersionProof"
    }
  }

  # ─────────────────────────────────────────────────────────────
  # (4) CUDA ENV PROOF NODE
  # ─────────────────────────────────────────────────────────────
  $cudaPath=$env:CUDA_PATH
  $pathBinOK=$false
  if($cudaPath){
    if($env:Path -like "*$cudaPath\bin*"){ $pathBinOK=$true }
  }

  if($status -eq "PASS_READY" -and (-not $cudaPath)){
    $status="FAIL_CUDA_PATH_MISSING"
    $smart += "STOP: CUDA_PATH missing after reboot."
    $smart += "Action: open System Properties → Environment Variables → set CUDA_PATH to CUDA root, reboot."
  }

  if($status -eq "PASS_READY" -and (-not $pathBinOK)){
    $status="FAIL_PATH_MISSING_BIN"
    $smart += "STOP: PATH does not include CUDA_PATH\bin after reboot."
    $smart += "Action: add %CUDA_PATH%\bin to USER PATH, reboot."
  }

  # ─────────────────────────────────────────────────────────────
  # (5) TORCH CUDA TRUTH NODE
  # ─────────────────────────────────────────────────────────────
  $torchCuda="NOT_RUN"
  if($status -eq "PASS_READY"){
    try{
      $torchCuda = python -c "import torch; print(torch.cuda.is_available())" 2>&1
    } catch {
      $torchCuda="TORCH_IMPORT_FAIL"
      $status="FAIL_TORCH_IMPORT"
      $smart += "STOP: python/torch import failed."
      $smart += "Action: verify python points to correct env, then pip install torch."
    }

    if($status -eq "PASS_READY" -and $torchCuda.Trim() -ne "True"){
      $status="FAIL_TORCH_CUDA_FALSE"
      $smart += "STOP: torch.cuda.is_available() is not True."
      $smart += "Action: install correct CUDA-enabled PyTorch build."
    }
  }

  # ─────────────────────────────────────────────────────────────
  # (6) EXTENSION COMPILE PROOF NODE (SANDBOX)
  # ─────────────────────────────────────────────────────────────
  $proofExe=$null
  if($status -eq "PASS_READY"){
    $sandbox = Join-Path $RepoDir "build\proof_$Tag"
    Ensure-Dir $sandbox
    Set-Location $sandbox

    @"
#include <stdio.h>
__global__ void hello() {}
int main(){
  hello<<<1,1>>>();
  printf("CUDA COMPILE PROOF OK\n");
  return 0;
}
"@ | Out-File "proof.cu" -Encoding UTF8 -Force

    $out = & $nvcc "proof.cu" "-o" "proof.exe" 2>&1
    Safe-Write $proofLog $out

    $proofExe = Join-Path $sandbox "proof.exe"
    if(-not (Test-Path $proofExe)){
      $status="FAIL_NVCC_COMPILE"
      $smart += "STOP: nvcc compile failed. See proof log."
      $smart += "Proof log → $proofLog"
    } else {
      $smart += "✅ NVCC COMPILE SUCCESS → $proofExe"
    }
  }

  # ─────────────────────────────────────────────────────────────
  # (7) ROOTMIRROR REQUIRED NODE
  # ─────────────────────────────────────────────────────────────
  $rm=@{state="SKIPPED";local_hash=$null;remote_hash=$null}
  if($status -eq "PASS_READY"){
    $rm = RootMirror-Verify
    if($rm.state -ne "VERIFIED"){
      $status="FAIL_ROOTMIRROR_NOT_VERIFIED"
      $smart += "STOP: RootMirror not VERIFIED (state=$($rm.state))."
      $smart += "Action: git push/pull until local==remote."
    } else {
      $smart += "✅ ROOTMIRROR VERIFIED"
    }
  }

  # ─────────────────────────────────────────────────────────────
  # (8) HASHED PASS TOKEN NODE (ONLY IF EVERYTHING PASSED)
  # ─────────────────────────────────────────────────────────────
  if($status -eq "PASS_READY"){
    $gitHash = $rm.local_hash
    $token = @"
PASS:$Tag
time=$(Get-Date -Format o)
git=$gitHash
nvcc=$nvcc
nvcc_version_proof=$nvccVersionProof
CUDA_PATH=$cudaPath
PATH_BIN_OK=$pathBinOK
torch_cuda=$($torchCuda.Trim())
proof_exe=$proofExe
reboot_required=$rebootRequired
reboot_confirmed=$rebootConfirmed
"@
    Safe-Write $passToken $token
    $status="PASS_UNLOCKED"
    $smart += "✅ PASS TOKEN MINTED → $passToken"
  }

  # ─────────────────────────────────────────────────────────────
  # ORACLE STATE
  # ─────────────────────────────────────────────────────────────
  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    status=$status
    SMART_FEEDBACK=@($smart)

    tokens=@{
      reboot_required=$rebootRequired
      reboot_confirmed=$rebootConfirmed
      pass_token=$passToken
    }

    nvcc=@{
      path=$nvcc
      version_proof=$nvccVersionProof
      version=$nvccVersion
      proof_log=$proofLog
      proof_exe=$proofExe
    }

    cuda_env=@{
      CUDA_PATH=$cudaPath
      PATH_BIN_OK=$pathBinOK
    }

    torch_cuda=$torchCuda

    rootmirror=$rm

    next_gate="Only if PASS_UNLOCKED → A2.0.V2.8.0 Batmobile CUDA Extension Engine"
  }

  Safe-Write $oraclePath  ($oracle | ConvertTo-Json -Depth 40)
  Safe-Write $reflectPath ($oracle | ConvertTo-Json -Depth 40)

  Print-Inline $oracle
  Write-Host "✅ $Tag COMPLETE (status=$status)" -ForegroundColor Green

}
catch{
  $err=$_.Exception.Message
  $fallback=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    status="FAIL_ORACLE_EXCEPTION"
    error=$err
  }
  if($oraclePath){
    $fallback | ConvertTo-Json -Depth 10 | Out-File $oraclePath -Encoding UTF8 -Force
  }
  Write-Host "❌ ORACLE EXCEPTION (state emitted): $err" -ForegroundColor Red
}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
