"""
Brute-force sanity check for the LSTM forward pass: recompute the gates,
cell state, and hidden state with a completely independent, unvectorized
implementation (explicit scalar loops, no numpy matrix-vector ops) and
confirm it matches lstm_step_forward exactly.
"""
import numpy as np
from lstm_scratch import lstm_step_forward

np.random.seed(3)
d, n_h = 2, 3
X = np.round(np.random.randn(4, d), 2)


def init(shape):
    return np.round(np.random.randn(*shape) * 0.3, 3)


Wf, Wi, Wg, Wo = [init((n_h, n_h + d)) for _ in range(4)]
bf, bi, bg, bo = [np.round(np.random.randn(n_h) * 0.1, 3) for _ in range(4)]


def sigmoid_scalar(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh_scalar(z):
    e_pos, e_neg = np.exp(z), np.exp(-z)
    return (e_pos - e_neg) / (e_pos + e_neg)


def brute_force_lstm_step(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo):
    n_h, total_dim = Wf.shape
    z = np.concatenate([h_prev, x_t])
    f_t = np.zeros(n_h); i_t = np.zeros(n_h); g_t = np.zeros(n_h); o_t = np.zeros(n_h)
    for k in range(n_h):
        af = bf[k]; ai = bi[k]; ag = bg[k]; ao = bo[k]
        for j in range(total_dim):
            af += Wf[k, j] * z[j]
            ai += Wi[k, j] * z[j]
            ag += Wg[k, j] * z[j]
            ao += Wo[k, j] * z[j]
        f_t[k] = sigmoid_scalar(af)
        i_t[k] = sigmoid_scalar(ai)
        g_t[k] = tanh_scalar(ag)
        o_t[k] = sigmoid_scalar(ao)
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * tanh_scalar(c_t)
    return h_t, c_t


h_prev_fast = np.zeros(n_h); c_prev_fast = np.zeros(n_h)
h_prev_brute = np.zeros(n_h); c_prev_brute = np.zeros(n_h)
max_diff = 0.0
for t in range(4):
    h_fast, c_fast, _ = lstm_step_forward(X[t], h_prev_fast, c_prev_fast,
                                           Wf, Wi, Wg, Wo, bf, bi, bg, bo)
    h_brute, c_brute = brute_force_lstm_step(X[t], h_prev_brute, c_prev_brute,
                                              Wf, Wi, Wg, Wo, bf, bi, bg, bo)
    diff = max(np.max(np.abs(h_fast - h_brute)), np.max(np.abs(c_fast - c_brute)))
    max_diff = max(max_diff, diff)
    print(f"t={t+1}: h (vectorized) = {np.round(h_fast,6)}  h (brute) = {np.round(h_brute,6)}  diff={diff:.2e}")
    h_prev_fast, c_prev_fast = h_fast, c_fast
    h_prev_brute, c_prev_brute = h_brute, c_brute

print(f"\nmax abs diff across all timesteps, gates and states: {max_diff:.2e}")
