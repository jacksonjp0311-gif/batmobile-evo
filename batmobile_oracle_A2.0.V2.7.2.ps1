param([string]$RepoDir,[string]$Tag)

$ErrorActionPreference="Stop"
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

# ───────────────────────────────────────────────
# NVCC HIGHEST VERSION NODE
# ───────────────────────────────────────────────
function Find-NVCC-Highest {
  $root="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
  if(-not(Test-Path $root)){ return $null }

  $candidates=@()
  Get-ChildItem $root -Directory | ForEach-Object{
    $nv=Join-Path $_.FullName "bin\nvcc.exe"
    if(Test-Path $nv){
      $candidates += $_.FullName
    }
  }

  if($candidates.Count -eq 0){ return $null }

  # Sort by folder name (v12.4 > v11.8)
  $best = $candidates | Sort-Object -Descending | Select-Object -First 1
  return (Join-Path $best "bin\nvcc.exe")
}

function Print-Inline($Obj){
  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  Write-Host "🜁 EXTENSION PROOF INLINE STATE"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 30
  Write-Host "══════════════════════════════════════════════════════════════"
}

try{
  Set-Location $RepoDir

  Ensure-Dir "artifacts"
  Ensure-Dir "build\proof_$Tag"

  $oraclePath  = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $passToken   = Join-Path $RepoDir "artifacts\PASS_$Tag.token"
  $proofLog    = Join-Path $RepoDir "artifacts\nvcc_proof_$Tag.txt"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — EXTENSION BUILD PROOF BEGIN" -ForegroundColor Cyan

  $smart=@()
  $status="PASS_READY"

  # CLEAN TREE REQUIRED
  if(Repo-IsDirty){
    $status="BLOCKED_DIRTY_TREE"
    $smart += "STOP: Repo dirty. Commit/stash before build proof."
  }

  # NVCC REQUIRED
  $nvcc = Find-NVCC-Highest
  if(-not $nvcc){
    $status="FAIL_NVCC_MISSING"
    $smart += "STOP: nvcc.exe missing. Install CUDA Toolkit Developer."
  }

  # TORCH CUDA REQUIRED
  $torchCuda = python -c "import torch; print(torch.cuda.is_available())"
  if($torchCuda.Trim() -ne "True"){
    $status="FAIL_TORCH_CUDA_FALSE"
    $smart += "STOP: torch.cuda.is_available() is not True."
  }

  # ───────────────────────────────────────────────
  # EXTENSION COMPILE PROOF NODE
  # ───────────────────────────────────────────────
  if($status -eq "PASS_READY"){
    $sandbox = Join-Path $RepoDir "build\proof_$Tag"
    Set-Location $sandbox

    # Minimal CUDA proof kernel
    @"
#include <stdio.h>
__global__ void hello() {}
int main(){
  hello<<<1,1>>>();
  printf("CUDA COMPILE PROOF OK\n");
  return 0;
}
"@ | Out-File "proof.cu" -Encoding UTF8 -Force

    $out = & $nvcc proof.cu -o proof.exe 2>&1
    $out | Out-File $proofLog -Encoding UTF8 -Force

    if(-not(Test-Path "proof.exe")){
      $status="FAIL_NVCC_COMPILE"
      $smart += "STOP: nvcc compile failed. See proof log."
    }
  }

  # ───────────────────────────────────────────────
  # HASHED PASS TOKEN NODE
  # ───────────────────────────────────────────────
  if($status -eq "PASS_READY"){
    $gitHash=(git rev-parse HEAD).Trim()
    $token="PASS:$Tag | git=$gitHash | nvcc=$nvcc | torch=$torchCuda"
    $token | Out-File $passToken -Encoding UTF8 -Force
    $status="PASS_UNLOCKED"
    $smart += "✅ EXTENSION BUILD PROOF SUCCESS"
    $smart += "✅ PASS TOKEN MINTED → $passToken"
  }

  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    status=$status
    SMART_FEEDBACK=@($smart)

    nvcc=$nvcc
    torch_cuda=$torchCuda
    proof_log=$proofLog

    next_gate="Only if PASS_UNLOCKED → A2.0.V2.8.0 Batmobile CUDA Extension Engine"
  }

  $oracle | ConvertTo-Json -Depth 30 | Out-File $oraclePath -Encoding UTF8 -Force

  Print-Inline $oracle
  Write-Host "✅ $Tag COMPLETE (status=$status)" -ForegroundColor Green

}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
