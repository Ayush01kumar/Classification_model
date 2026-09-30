"""
Brute-force sanity check for conv2d_forward: recompute the same
convolution with a completely independent quadruple-nested-loop
implementation (no slicing, no vectorized patch multiply) and confirm
it matches conv2d_forward exactly. Also cross-checks against
scipy.signal.correlate2d in 'valid' mode, since frameworks implement
"convolution" as cross-correlation (no kernel flip), same as this
series' scratch code.
"""
import numpy as np
from scipy.signal import correlate2d
from cnn_scratch import conv2d_forward

np.random.seed(1)
X = np.round(np.random.randn(6, 6), 2)
K = np.round(np.random.randn(3, 3), 2)
b = 0.25


def brute_force_conv(X, K, b):
    H, W = X.shape
    kh, kw = K.shape
    H_out, W_out = H - kh + 1, W - kw + 1
    Z = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += X[i + u, j + v] * K[u, v]
            Z[i, j] = total + b
    return Z


Z_fast = conv2d_forward(X, K, b)
Z_brute = brute_force_conv(X, K, b)
Z_scipy = correlate2d(X, K, mode="valid") + b

print("conv2d_forward:\n", Z_fast)
print("\nbrute-force quadruple loop:\n", Z_brute)
print("\nscipy.signal.correlate2d (valid) + b:\n", Z_scipy)

print("\nmax abs diff (scratch vs brute-force):", np.max(np.abs(Z_fast - Z_brute)))
print("max abs diff (scratch vs scipy):", np.max(np.abs(Z_fast - Z_scipy)))
