"""
Hand-worked example: run the LSTM forward and backward pass on the SAME
4-month rising-utilization sequence used in the companion RNN article,
so the two architectures can be compared directly on identical input.
Prints every gate, the cell state, the hidden state, and every gradient.
"""
import numpy as np
import pandas as pd
from lstm_scratch import LSTM

np.set_printoptions(precision=4, suppress=True)

df = pd.read_csv("data/hand_example.csv")
X = df[["utilization"]].values
print("Input sequence X (utilization ratio per month):")
print(X.flatten())
print()

n_h = 2
Wf = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
Wi = np.array([[0.3, 0.1, 0.2], [0.1, 0.3, 0.2]])
Wg = np.array([[0.2, 0.2, 0.5], [0.3, 0.1, 0.4]])
Wo = np.array([[0.4, 0.1, 0.2], [0.2, 0.3, 0.1]])
bf = np.zeros(n_h); bi = np.zeros(n_h); bg = np.zeros(n_h); bo = np.zeros(n_h)
W_out = np.array([[0.6, -0.3], [0.2, 0.5]])
b_out = np.zeros(2)

model = LSTM(Wf, Wi, Wg, Wo, bf, bi, bg, bo, W_out, b_out)
probs, cache = model.forward(X)

print("Per-timestep gates and states:")
for t in range(4):
    c = cache["caches"][t]
    print(f"  t={t+1}: f={c['f_t']}  i={c['i_t']}  g={c['g_t']}  o={c['o_t']}  "
          f"c_t={cache['cs'][t]}  h_t={cache['hs'][t]}")
print()

print("Final hidden state h_T:", cache["hs"][-1])
print("Final cell state c_T:", cache["cs"][-1])
print("Logits:", cache["logits"])
print("Softmax probabilities:", probs)
print()

y_onehot = np.array([0., 1.])
loss = -np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))
print("True label (one-hot):", y_onehot)
print("Cross-entropy loss:", loss)
print()

grads = model.backward(cache, y_onehot)
print("dL/dlogits = p - y:", probs - y_onehot)
print()
print("Gradient norms flowing into each timestep (BPTT), t=T..1:")
for t in sorted(grads["dh_norms"].keys(), reverse=True):
    print(f"  t={t+1}: ||dh_t|| = {grads['dh_norms'][t]:.6f}   ||dc_t|| = {grads['dc_norms'][t]:.6f}")
print()
print("dL/dWf:\n", grads["dWf"])
print("dL/dWi:\n", grads["dWi"])
print("dL/dWg:\n", grads["dWg"])
print("dL/dWo:\n", grads["dWo"])
print("dL/dW_out:\n", grads["dW_out"])
