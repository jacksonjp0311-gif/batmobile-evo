# Batmobile.EVO — Evolution Cleanup + Benchmark Plan

## Context snapshot
- The repo includes benchmark scripts under `benchmarks/` that currently print results but do not emit results JSON artifacts or write the `logs/` + `artifacts/latest_*.txt` pointers described in `docs/OUTPUT_BOOTSTRAP.md`.【F:benchmarks/bench_spherical_harmonics.py†L1-L200】【F:docs/OUTPUT_BOOTSTRAP.md†L1-L8】
- The existing harness expects `benchmarks/microbench/*` and `benchmarks/end2end/*` files that are not present in the tree, so it will reliably log `FAILED_BUILD` for each missing bench on any run.【F:benchmarks/harness/run_harness.py†L24-L121】
- The benchmark contract requires results JSON to conform to `benchmarks/results_schema.json` and the falsification plan expects standard failure-state annotations in those artifacts.【F:benchmarks/results_schema.json†L1-L24】【F:docs/hypothesis/falsification_plan.md†L1-L16】
- Canon lock notes that evolution is blocked until a PASS token exists, and extension builds are currently not allowed until the repo is unblocked and `nvcc` is installed.【F:docs/canon/TRUTH_PROTOCOL_CANON.md†L1-L7】

## Goal
Finish the current evolution phase by turning benchmarks into a reproducible, contract-valid pipeline that produces audit-ready artifacts (logs + JSON results) suitable for posting public benchmark results.

## Recommended execution plan
### 1) Clean up benchmark inventory + choose source of truth
- **Align harness with reality**: either (a) create the missing `microbench/` + `end2end/` scripts, or (b) update `benchmarks/harness/run_harness.py` to call the existing scripts in `benchmarks/` (e.g., `bench_spherical_harmonics.py`, `benchmark_tensor_product.py`, `benchmark_fused_sh_tp.py`, `benchmark_e2e_mace.py`).【F:benchmarks/harness/run_harness.py†L24-L121】【F:README.md†L75-L88】
- **Define the benchmark matrix**: choose the canonical set of microbench + end-to-end benchmarks that will be reported publicly, and codify that list in a single manifest (e.g., a small `benchmarks/bench_registry.json` read by the harness).

### 2) Enforce artifact contract + output bootstrap
- **Make every benchmark emit JSON** matching `benchmarks/results_schema.json`, including environment fingerprint and run metadata (exit code, timing, stdout/stderr paths).【F:benchmarks/results_schema.json†L1-L24】
- **Add output bootstrap hooks** so every run also writes:
  - `logs/run_<TAG>.log`
  - `artifacts/latest_<TAG>.txt` pointing at the most recent JSON
  per `docs/OUTPUT_BOOTSTRAP.md`.【F:docs/OUTPUT_BOOTSTRAP.md†L1-L8】
- **Schema validation**: add a lightweight schema check (even a custom validator) in the harness to mark `FAILED_CONTRACT` when a result file is missing required fields.

### 3) Resolve build/run gating to unlock the PASS token
- **Confirm CUDA toolchain availability** (per canon lock) and document the expected environment (CUDA version, GPU, PyTorch version).【F:docs/canon/TRUTH_PROTOCOL_CANON.md†L1-L7】
- **Decide on the PASS token rule**: e.g., all benchmarks in the registry produce `OK` status + contract-valid JSON + reproducible tolerance within N runs. This should be written down in a single place (new doc or existing canon note).

### 4) Reproducibility + publishing workflow
- **Pin dependencies** for benchmarking runs (requirements file or lockfile) and capture them in the environment fingerprint; the harness already collects `pip freeze` output, so ensure it is preserved in the JSON artifact.【F:benchmarks/harness/run_harness.py†L39-L70】
- **Automate a “publish bundle”**: a script that packages the JSON results, logs, and a human-readable summary for posting.
- **Add a README section** describing exactly how to run the harness + where artifacts are written to avoid ambiguity for external reviewers.

## Expected deliverables (minimum)
1. A single benchmark harness that runs **only existing** benchmarks and produces contract-valid JSON artifacts.
2. Output bootstrap files for every run (logs + latest pointer).
3. A short doc section that defines PASS token criteria + what “reproducible” means for this project.

## Suggested next actions (short-term)
1. Edit `benchmarks/harness/run_harness.py` to reference the actual benchmark scripts in `benchmarks/` or create the missing `microbench/` structure to match it.【F:benchmarks/harness/run_harness.py†L24-L121】
2. Add a thin results wrapper for each benchmark script to emit schema-valid JSON and write output bootstrap files.【F:benchmarks/results_schema.json†L1-L24】【F:docs/OUTPUT_BOOTSTRAP.md†L1-L8】
3. Document PASS token criteria in the canon or a new dedicated file and wire it into CI or a local runner.
