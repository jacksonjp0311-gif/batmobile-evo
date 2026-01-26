# BATMOBILE.EVO — Falsification Plan (A2.0.V1.2)

This is an execution-level scaffold: it specifies *failure conditions*.

## Fail States (examples)
- **FAILED_BUILD**: cannot compile or import.
- **FAILED_REPRO**: cannot reproduce metrics within tolerance under pinned env.
- **FAILED_CONTRACT**: output does not match schema.
- **FAILED_DRIFT**: unstable results under small perturbations (seed changes, batch order).

## Required Artifacts
- Environment snapshot: Python version, OS, CUDA/CPU, dependency lock.
- Result JSON matching enchmarks/results_schema.json.
