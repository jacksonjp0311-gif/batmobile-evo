# BATMOBILE.EVO — Benchmark Contract (A2.0.V1.2)

## Purpose
Define *how* results may be produced and interpreted without implying correctness or performance.

## Contract
- All runs MUST emit enchmarks/results_*.json conforming to enchmarks/results_schema.json.
- Each result must include:
  - timestamp (UTC), environment fingerprint, task name, metrics dict, notes.

## Non-Claims
- Benchmarks do not prove generalization, scaling, or optimality.
- Any reported numbers are context-bound to environment + task.

## Falsification
- If results are not reproducible under pinned dependencies + fixed seed, mark as **FAILED_REPRO**.
