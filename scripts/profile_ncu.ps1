# BATMOBILE.EVO — Nsight Compute hook (A2.0.V1.3)
# Optional. Artifact capture only.

param(
  [string]$Python = "python",
  [ValidateSet("microbench_tensor_product","microbench_spherical_harmonics","microbench_neighbor_list","end2end_message_passing")]
  [string]$Target = "microbench_tensor_product"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Split-Path -Parent (Split-Path -Parent $($MyInvocation.MyCommand.Path)))
Set-Location -LiteralPath ""

function Find-Ncu {
  $(Get-Command ncu -ErrorAction SilentlyContinue)?.Source
}

$ncu = Find-Ncu
if (-not $ncu) {
  Write-Host "⚠ ncu not found in PATH. Install Nsight Compute and re-run." -ForegroundColor Yellow
  exit 0
}

$map = @{
  "microbench_spherical_harmonics" = "benchmarks/microbench/bench_spherical_harmonics.py"
  "microbench_tensor_product"      = "benchmarks/microbench/bench_tensor_product.py"
  "microbench_neighbor_list"       = "benchmarks/microbench/bench_neighbor_list.py"
  "end2end_message_passing"        = "benchmarks/end2end/bench_message_passing.py"
}

$rel = $map[$Target]
$bench = Join-Path $RepoRoot $rel
if (-not (Test-Path -LiteralPath $bench)) {
  throw "Benchmark missing: $bench"
}

$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$out = Join-Path $RepoRoot ("benchmarks/profiles/ncu_{0}_{1}" -f $Target, $stamp)

Write-Host "𓇳 Profiling: $Target"
Write-Host "🜂 Output: $out"

# capture a reasonable default set; user can refine
& ncu --set full --target-processes all -o $out $Python $bench

Write-Host "🜁 Nsight profile captured." -ForegroundColor Green
