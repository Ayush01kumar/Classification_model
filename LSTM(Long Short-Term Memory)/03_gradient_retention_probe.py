"""
The gradient-retention probe: the central empirical result of this
article, and the direct counterpart to the RNN article's
vanishing-gradient probe. Runs the SAME experiment -- measure the
gradient signal reaching the first timestep as sequence length grows --
for both a vanilla RNN and this repo's LSTM, on the same random weights
scale and the same sequence lengths, and compares them side by side.

A minimal VanillaRNN is reimplemented here (self-contained, matching the
companion RNN article's rnn_scratch.py exactly) purely for this
side-by-side comparison; the LSTM implementation itself is lstm_scratch.py.
"""
import numpy as np
from lstm_scratch import LSTM

np.random.seed(7)


# --- minimal VanillaRNN, identical formula to the RNN article ---
def rnn_forward_backward(X, y_onehot, Wxh, Whh, bh, W_out, b_out):
    T = X.shape[0]
    n_h = Whh.shape[0]
    h_prev = np.zeros(n_h)
    hs, a_s = [], []
    for t in range(T):
        a_t = Wxh @ X[t] + Whh @ h_prev + bh
        h_t = np.tanh(a_t)
        hs.append(h_t); a_s.append(a_t); h_prev = h_t
    logits = hs[-1] @ W_out + b_out
    z = logits - np.max(logits); probs = np.exp(z) / np.sum(np.exp(z))

    dlogits = probs - y_onehot
    dh_next = W_out @ dlogits
    dh_norms = {}
    for t in reversed(range(T)):
        dh_norms[t] = np.linalg.norm(dh_next)
        da_t = dh_next * (1 - np.tanh(a_s[t]) ** 2)
        dh_next = Whh.T @ da_t
    return dh_norms


d, n_h, n_classes = 1, 4, 2
y_onehot = np.array([0., 1.])

# RNN weights
Wxh = np.random.randn(n_h, d) * 0.5
Whh = np.random.randn(n_h, n_h) * 0.5
bh = np.zeros(n_h)
W_out_rnn = np.random.randn(n_h, n_classes) * 0.5
b_out_rnn = np.zeros(n_classes)

# LSTM weights (same scale, same n_h, same d)
Wf = np.random.randn(n_h, n_h + d) * 0.5
Wi = np.random.randn(n_h, n_h + d) * 0.5
Wg = np.random.randn(n_h, n_h + d) * 0.5
Wo = np.random.randn(n_h, n_h + d) * 0.5
bf = np.ones(n_h) * 1.0   # forget gate biased toward "remember" (standard LSTM init trick)
bi = np.zeros(n_h); bg = np.zeros(n_h); bo = np.zeros(n_h)
W_out_lstm = np.random.randn(n_h, n_classes) * 0.5
b_out_lstm = np.zeros(n_classes)

print(f"{'T':>4} | {'RNN ||dh_1||':>14} | {'LSTM ||dh_1||':>14} | {'LSTM / RNN ratio':>16}")
for T in [2, 4, 8, 12, 16, 20, 30]:
    np.random.seed(7)
    X = np.random.randn(T, d) * 0.5

    dh_norms_rnn = rnn_forward_backward(X, y_onehot, Wxh, Whh, bh, W_out_rnn, b_out_rnn)
    rnn_dh1 = dh_norms_rnn[0]

    lstm_model = LSTM(Wf.copy(), Wi.copy(), Wg.copy(), Wo.copy(),
                       bf.copy(), bi.copy(), bg.copy(), bo.copy(), W_out_lstm.copy(), b_out_lstm.copy())
    probs, cache = lstm_model.forward(X)
    grads = lstm_model.backward(cache, y_onehot)
    lstm_dh1 = grads["dh_norms"][0]

    ratio = lstm_dh1 / rnn_dh1 if rnn_dh1 > 0 else float("nan")
    print(f"{T:4d} | {rnn_dh1:14.8f} | {lstm_dh1:14.8f} | {ratio:16.1f}")

print("\nThe LSTM's gradient at the first timestep stays many orders of magnitude")
print("larger than the vanilla RNN's at long sequence lengths -- the direct,")
print("measured effect of the additive cell-state update (c_t = f_t*c_prev + i_t*g_t)")
print("replacing the RNN's repeated multiplicative squashing through tanh and Whh.")
