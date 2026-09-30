"""
Brute-force sanity check for the RNN forward pass: recompute the
per-timestep hidden states with a completely independent, unvectorized
implementation (explicit dot products via python loops, no numpy
matrix-vector ops) and confirm it matches rnn_step_forward exactly.
"""
import numpy as np
from rnn_scratch import rnn_step_forward

np.random.seed(2)
d, n_h, T = 3, 4, 5
X = np.round(np.random.randn(T, d), 2)
Wxh = np.round(np.random.randn(n_h, d) * 0.3, 3)
Whh = np.round(np.random.randn(n_h, n_h) * 0.3, 3)
bh = np.round(np.random.randn(n_h) * 0.1, 3)


def tanh_scalar(z):
    e_pos = np.exp(z)
    e_neg = np.exp(-z)
    return (e_pos - e_neg) / (e_pos + e_neg)


def brute_force_rnn_step(x_t, h_prev, Wxh, Whh, bh):
    n_h, d = Wxh.shape
    h_t = np.zeros(n_h)
    for i in range(n_h):
        total = bh[i]
        for j in range(d):
            total += Wxh[i, j] * x_t[j]
        for j in range(n_h):
            total += Whh[i, j] * h_prev[j]
        h_t[i] = tanh_scalar(total)
    return h_t


h_prev_fast = np.zeros(n_h)
h_prev_brute = np.zeros(n_h)
max_diff = 0.0
for t in range(T):
    h_fast, _ = rnn_step_forward(X[t], h_prev_fast, Wxh, Whh, bh)
    h_brute = brute_force_rnn_step(X[t], h_prev_brute, Wxh, Whh, bh)
    diff = np.max(np.abs(h_fast - h_brute))
    max_diff = max(max_diff, diff)
    print(f"t={t+1}: rnn_step_forward = {np.round(h_fast, 6)}  "
          f"brute-force = {np.round(h_brute, 6)}  diff = {diff:.2e}")
    h_prev_fast = h_fast
    h_prev_brute = h_brute

print(f"\nmax abs diff across all timesteps: {max_diff:.2e}")
