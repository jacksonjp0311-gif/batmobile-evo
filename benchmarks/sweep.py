import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json, platform, time
from datetime import datetime
from src.tensor_product.kernel_stub import tensor_product_stub

def main():
    results = []
    for ell in range(0, 9):
        r = tensor_product_stub(ell=ell)
        results.append(r)

    artifact = {
        "contract": "batmobile.next.sweep",
        "version": "A2.0.V1.7",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version()
        },
        "sweep": results
    }

    out = f"benchmarks/artifacts/sweep_{int(time.time())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("[BATMOBILE] Sweep artifact written →", out)

if __name__ == "__main__":
    main()