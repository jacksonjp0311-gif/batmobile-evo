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
  Write-Host "🜁 SMART CUDA ORACLE STATE (COPY/PASTE READY)"
  Write-Host "══════════════════════════════════════════════════════════════"
  $Obj | ConvertTo-Json -Depth 12
  Write-Host "══════════════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
# VERIFIER NODE 1 — DISK NVCC SCAN
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

try{

  Set-Location $RepoDir
  Ensure-Dir "artifacts"

  $oraclePath = Join-Path $RepoDir "artifacts\oracle_$Tag.json"

  Write-Host ""
  Write-Host "𓇳 BATMOBILE $Tag — SMART CUDA TOOLKIT ORACLE BEGIN" -ForegroundColor Cyan

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 1 — Backend Truth
  # ───────────────────────────────────────────────────────────────────────────
  $nvccDisk   = Find-NVCC
  $cudaPath   = $env:CUDA_PATH
  $torchCuda  = "UNKNOWN"

  try{
    $torchCuda = python -c "import torch; print(torch.cuda.is_available())"
  } catch {
    $torchCuda = "TORCH_NOT_FOUND"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 2 — Gate Classification + Smart Feedback
  # ───────────────────────────────────────────────────────────────────────────
  $status="OK"
  $guidance=@()

  if($torchCuda -match "True"){
    $guidance += "Torch CUDA runtime: TRUE (GPU driver working)"
  }

  if($nvccDisk.Count -eq 0){
    $status="CUDA_TOOLKIT_MISSING"
    $guidance += "nvcc.exe NOT FOUND → CUDA Toolkit NOT installed"
    $guidance += ""
    $guidance += "NEXT ACTION REQUIRED:"
    $guidance += "1) Download CUDA Toolkit (not just driver)"
    $guidance += "2) Install with 'CUDA Compiler (nvcc)' checked"
    $guidance += "3) Reboot Windows"
    $guidance += "4) Rerun Oracle A2.0.V2.6.2"
    $guidance += ""
    $guidance += "Expected nvcc path:"
    $guidance += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\nvcc.exe"
  }

  if($status -eq "OK" -and -not $cudaPath){
    $status="CUDA_PATH_MISSING"
    $guidance += "CUDA Toolkit exists but CUDA_PATH env var missing"
    $guidance += "Fix by reinstalling toolkit or setting CUDA_PATH manually"
  }

  # ───────────────────────────────────────────────────────────────────────────
  # SECTION 3 — Oracle State Emit
  # ───────────────────────────────────────────────────────────────────────────
  $oracle=@{
    tag=$Tag
    time=(Get-Date).ToString("o")
    status=$status

    torch_cuda=$torchCuda

    nvcc=@{
      disk_candidates=@($nvccDisk)
      primary=$(if($nvccDisk.Count -gt 0){$nvccDisk[0]}else{$null})
    }

    cuda_env=@{
      CUDA_PATH=$cudaPath
    }

    guidance=@($guidance)

    next_gate="If NVCC OK → A2.0.V2.7.0 Extension Build Proof Node"
  }

  $oracle | ConvertTo-Json -Depth 12 |
    Out-File -FilePath $oraclePath -Encoding UTF8 -Force

  Print-Inline $oracle

  Write-Host ""
  Write-Host "🜁 Oracle saved → $oraclePath" -ForegroundColor Green
  Write-Host "✅ SMART CUDA ORACLE COMPLETE" -ForegroundColor Green

}
finally{
  Set-Location $StartDir
  Write-Host "𓂀 RETURN TO ROOT." -ForegroundColor Cyan
}
