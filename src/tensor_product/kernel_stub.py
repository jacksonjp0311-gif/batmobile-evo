import time
import numpy as np

L_SPECIALIZE = 4

def tensor_product_stub(n=256, ell=2):
    a = np.random.randn(n)
    b = np.random.randn(n)

    t0 = time.time()
    c = np.dot(a, b)
    dt = (time.time() - t0) * 1000.0

    return {
        "ell": ell,
        "specialized": bool(ell <= L_SPECIALIZE),
        "dot": float(c),
        "ms": float(dt)
    }