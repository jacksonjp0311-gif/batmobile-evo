import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json, platform, time
from datetime import datetime
from src.tensor_product.kernel_stub import tensor_product_stub

def main():
    result = tensor_product_stub()

    artifact = {
        "contract": "batmobile.next.benchmarks",
        "version": "A2.0.V1.6.3",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version()
        },
        "metrics": result
    }

    out = f"benchmarks/artifacts/results_{int(time.time())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("[BATMOBILE] Artifact written →", out)

if __name__ == "__main__":
    main()