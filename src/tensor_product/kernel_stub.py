import time
import numpy as np

from src.tensor_product.specialized.tp_l0 import tp as tp0
from src.tensor_product.specialized.tp_l1 import tp as tp1
from src.tensor_product.specialized.tp_l2 import tp as tp2
from src.tensor_product.specialized.tp_l3 import tp as tp3
from src.tensor_product.specialized.tp_l4 import tp as tp4

L_SPECIALIZE = 4

def tensor_product_stub(n=256, ell=2):

    t0 = time.time()

    if ell == 0: val = tp0(n)
    elif ell == 1: val = tp1(n)
    elif ell == 2: val = tp2(n)
    elif ell == 3: val = tp3(n)
    elif ell == 4: val = tp4(n)
    else:
        a = np.random.randn(n)
        b = np.random.randn(n)
        val = float(np.dot(a,b))

    dt = (time.time() - t0) * 1000.0

    return {
        "ell": ell,
        "specialized": bool(ell <= L_SPECIALIZE),
        "dot": float(val),
        "ms": float(dt)
    }