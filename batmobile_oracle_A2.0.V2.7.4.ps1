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
  Write-Host "🜁 CUDA ENV REPAIR INLINE STATE"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 30
  Write-Host "══════════════════════════════════════════════════════════════"
}

try{
  Set-Location $RepoDir
  Ensure-Dir "artifacts"

  $oraclePath = Join-Path $RepoDir "artifacts\oracle_$Tag.json"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — CUDA ENV AUTO-REPAIR BEGIN" -ForegroundColor Cyan

  $smart=@()
  $status="PASS_READY"

  # NVCC REQUIRED
  $nvcc = Find-NVCC
  if(-not $nvcc){
    $status="FAIL_NVCC_MISSING"
    $smart += "STOP: nvcc.exe missing. Toolkit not installed."
  }

  # CUDA ROOT EXTRACTION
  $cudaRoot=$null
  if($nvcc){
    $cudaRoot = Split-Path (Split-Path $nvcc -Parent) -Parent
    $smart += "NVCC FOUND → $nvcc"
    $smart += "CUDA ROOT → $cudaRoot"
  }

  # CUDA_PATH AUTOSET
  if($status -eq "PASS_READY"){
    if(-not $env:CUDA_PATH){
      [Environment]::SetEnvironmentVariable("CUDA_PATH",$cudaRoot,"User")
      $smart += "✅ CUDA_PATH SET (User) → $cudaRoot"
    } else {
      $smart += "CUDA_PATH already exists → $env:CUDA_PATH"
    }
  }

  # PATH BIN PATCH
  if($status -eq "PASS_READY"){
    $binPath = Join-Path $cudaRoot "bin"
    $userPath = [Environment]::GetEnvironmentVariable("Path","User")

    if($userPath -notlike "*$binPath*"){
      $newPath = "$userPath;$binPath"
      [Environment]::SetEnvironmentVariable("Path",$newPath,"User")
      $smart += "✅ PATH PATCHED (User) → +CUDA\\bin"
    } else {
      $smart += "PATH already contains CUDA\\bin"
    }

    $status="REBOOT_REQUIRED"
    $smart += "⚠ REBOOT REQUIRED before compile proof."
    $smart += "After reboot → run A2.0.V2.7.5 Compile Proof Node."
  }

  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    status=$status
    SMART_FEEDBACK=@($smart)

    nvcc=$nvcc
    cuda_root=$cudaRoot

    next_gate="Only if reboot complete → A2.0.V2.7.5 Compile Proof Node"
  }

  $oracle | ConvertTo-Json -Depth 30 | Out-File $oraclePath -Encoding UTF8 -Force

  Print-Inline $oracle
  Write-Host "✅ $Tag COMPLETE (status=$status)" -ForegroundColor Green

}
finally{
  Set-Location $RepoDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
