import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json, platform, time, statistics
from datetime import datetime, timezone
from src.tensor_product.kernel_stub import tensor_product_stub

ELL_VALUES = list(range(0, 9))
N_VALUES   = [128, 256, 512, 1024]
TRIALS     = 5

def main():
    grid = {}

    for n in N_VALUES:
        for ell in ELL_VALUES:
            ms_runs = []
            for _ in range(TRIALS):
                r = tensor_product_stub(n=n, ell=ell)
                ms_runs.append(r["ms"])

            key = f"n={n},ell={ell}"
            grid[key] = {
                "n": n,
                "ell": ell,
                "trials": TRIALS,
                "mean_ms": statistics.mean(ms_runs),
                "std_ms": statistics.pstdev(ms_runs),
                "specialized": bool(ell <= 4)
            }

    artifact = {
        "contract": "batmobile.next.grid_sweep",
        "version": "A2.0.V1.9",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version()
        },
        "parameters": {
            "ell_values": ELL_VALUES,
            "n_values": N_VALUES,
            "trials": TRIALS
        },
        "grid_results": grid
    }

    out = f"benchmarks/artifacts/grid_{int(time.time())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("[BATMOBILE] Grid artifact written ->", out)

if __name__ == "__main__":
    main()