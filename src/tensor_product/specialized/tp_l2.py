import numpy as np
def tp(n=256):
    a = np.random.randn(n)
    b = np.random.randn(n)
    return float(np.dot(a,b))