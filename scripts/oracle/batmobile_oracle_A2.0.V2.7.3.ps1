param([string]$RepoDir,[string]$Tag)

$ErrorActionPreference="Stop"
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

function Ensure-Dir($p){
  if(-not(Test-Path $p)){
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

function Find-NVCC {
  $root="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
  if(-not(Test-Path $root)){ return $null }

  $nvcc = Get-ChildItem $root -Recurse -Filter nvcc.exe -ErrorAction SilentlyContinue |
          Select-Object -First 1

  if($nvcc){ return $nvcc.FullName }
  return $null
}

function Print-Inline($Obj){
  Write-Host ""
  Write-Host "══════════════════════════════════════════════════════════════"
  Write-Host "🜁 CUDA INSTALL ORACLE INLINE STATE"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 30
  Write-Host "══════════════════════════════════════════════════════════════"
}

try{
  Set-Location $RepoDir
  Ensure-Dir "artifacts"

  $oraclePath = Join-Path $RepoDir "artifacts\oracle_$Tag.json"
  $proofTxt   = Join-Path $RepoDir "artifacts\nvcc_version_$Tag.txt"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — CUDA INSTALL VERIFICATION BEGIN" -ForegroundColor Cyan

  $smart=@()
  $status="PASS_READY"

  # NVCC DISK NODE
  $nvcc = Find-NVCC
  if(-not $nvcc){
    $status="FAIL_NVCC_MISSING"
    $smart += "STOP: nvcc.exe not found."
    $smart += "DOWNLOAD CUDA TOOLKIT HERE:"
    $smart += "https://developer.nvidia.com/cuda-downloads"
    $smart += "Install → Developer Installer → Check 'CUDA Compiler (nvcc)'"
  }

  # CUDA_PATH NODE
  $cudaPath=$env:CUDA_PATH
  if($status -eq "PASS_READY" -and -not $cudaPath){
    $status="FAIL_CUDA_PATH_MISSING"
    $smart += "STOP: CUDA_PATH env var missing."
    $smart += "Fix: reboot after install or reinstall toolkit."
  }

  # PATH BIN NODE
  $pathOK=$false
  if($cudaPath){
    if($env:Path -like "*$cudaPath\\bin*"){ $pathOK=$true }
  }
  if($status -eq "PASS_READY" -and -not $pathOK){
    $status="FAIL_PATH_MISSING_BIN"
    $smart += "STOP: PATH does not include CUDA\\bin."
    $smart += "Fix: reboot or add CUDA bin manually."
  }

  # NVIDIA-SMI NODE
  $smi="NOT_RUN"
  try{
    $smi = nvidia-smi 2>&1
  } catch {
    $smi = "FAIL"
  }
  if($status -eq "PASS_READY" -and $smi -eq "FAIL"){
    $status="FAIL_NVIDIA_SMI"
    $smart += "STOP: nvidia-smi failed. GPU driver not installed."
  }

  # VERSION PROOF NODE
  $nvccVersion="NOT_RUN"
  if($status -eq "PASS_READY"){
    $nvccVersion = & $nvcc --version 2>&1
    $nvccVersion | Out-File $proofTxt -Encoding UTF8 -Force
    $smart += "✅ CUDA TOOLKIT VERIFIED"
    $smart += "Next Gate Unlocked → A2.0.V2.7.4 Extension Build Proof"
    $status="PASS_VERIFIED"
  }

  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    status=$status
    SMART_FEEDBACK=@($smart)

    nvcc=$nvcc
    cuda_env=@{
      CUDA_PATH=$cudaPath
      PATH_BIN_OK=$pathOK
    }

    nvidia_smi=$smi
    version_proof=$proofTxt

    next_gate="Only if PASS_VERIFIED → A2.0.V2.7.4 Compile Proof Node"
  }

  $oracle | ConvertTo-Json -Depth 30 | Out-File $oraclePath -Encoding UTF8 -Force

  Print-Inline $oracle
  Write-Host "✅ $Tag COMPLETE (status=$status)" -ForegroundColor Green

}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
