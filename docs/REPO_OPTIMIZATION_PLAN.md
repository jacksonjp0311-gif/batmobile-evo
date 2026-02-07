# Batmobile.EVO — Repo Optimization + Evolution Additions Plan

## Scope
This plan focuses on:
- reducing root-level clutter while preserving access to important artifacts,
- tightening benchmark reproducibility and PASS-token requirements,
- clarifying project documentation for contributors and benchmark reviewers.

## Current observations
- Root contains a mix of build/config files, benchmark images, and multiple PowerShell oracle scripts that can be grouped into dedicated directories for clarity.
- Benchmark harness outputs contract-valid JSON and audit logs, but individual benchmark scripts still print results without emitting per-benchmark JSON artifacts.
- PASS token criteria are required by canon but not yet codified in a single document.

## Optimization plan
### 1) Root cleanup + organization
- Move `batmobile_oracle_*.ps1` into `scripts/oracle/`.
- Move benchmark images into `docs/assets/`.
- Keep only essential top-level files: `README.md`, `LICENSE`, `pyproject.toml`, `setup.py`, `CONTRIBUTING.md`, and core directories.

### 2) Benchmark artifact completeness
- Add a small benchmark result writer helper (e.g., `benchmarks/harness/write_result.py`) so each benchmark script can emit a per-benchmark JSON file that conforms to `benchmarks/results_schema.json`.
- Update benchmark scripts to call the helper and include:
  - benchmark name, timestamp, and environment fingerprint
  - metrics dictionary
  - status (`OK` / `FAILED_*`) and notes

### 3) PASS token criteria
- Define a PASS token policy in a canonical doc (e.g., `docs/canon/PASS_TOKEN.md`):
  - all benchmarks in `bench_registry.json` must run
  - harness schema validation must pass
  - reproducibility check (N repeated runs within tolerance)
- Add a simple harness flag (or separate script) to execute and verify PASS conditions.

### 4) Reproducible benchmark environment
- Add a pinned benchmark env file (e.g., `requirements-bench.txt` or `environment.yml`).
- Document the tested CUDA + PyTorch stack in README and the environment file.

### 5) Publishing bundle for benchmark review
- Add a `scripts/package_bench_results.py` to collect:
  - `benchmarks/results/results_<stamp>.json`
  - `logs/run_<stamp>.log`
  - `artifacts/latest_<stamp>.txt`
  - environment snapshot (`pip freeze`)
- Output a single zip/tarball for easy review or posting.

## Near-term execution checklist
1) Update benchmark scripts to emit JSON artifacts.
2) Add PASS token doc + verification flag.
3) Create pinned benchmark environment spec.
4) Ship a benchmark bundling script.

## Notes
- These items keep the repo aligned with the benchmark contract and canon evolution rules while reducing friction for external reviewers.
