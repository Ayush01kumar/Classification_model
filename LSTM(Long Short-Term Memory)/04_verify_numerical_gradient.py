"""
Verify LSTM.backward() (full BPTT through gates + cell state) against
the numerical (finite-difference) gradient checker, for every parameter
(Wf, Wi, Wg, Wo, bf, bi, bg, bo, W_out, b_out) and the input X.
"""
import numpy as np
from lstm_scratch import LSTM, numerical_gradient

np.random.seed(0)
np.set_printoptions(precision=6, suppress=True)

T, d, n_h, n_classes = 5, 3, 4, 2
X = np.round(np.random.randn(T, d) * 0.5, 3)


def init(shape):
    return np.random.randn(*shape) * 0.3


Wf, Wi, Wg, Wo = [init((n_h, n_h + d)) for _ in range(4)]
bf, bi, bg, bo = [np.random.randn(n_h) * 0.1 for _ in range(4)]
W_out = init((n_h, n_classes))
b_out = np.random.randn(n_classes) * 0.1
y_onehot = np.array([0., 1.])

model = LSTM(Wf.copy(), Wi.copy(), Wg.copy(), Wo.copy(),
             bf.copy(), bi.copy(), bg.copy(), bo.copy(), W_out.copy(), b_out.copy())


def loss_fn():
    probs, _ = model.forward(X)
    return -np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))


probs, cache = model.forward(X)
grads = model.backward(cache, y_onehot)

checks = [
    ("Wf", model.Wf, grads["dWf"]), ("Wi", model.Wi, grads["dWi"]),
    ("Wg", model.Wg, grads["dWg"]), ("Wo", model.Wo, grads["dWo"]),
    ("bf", model.bf, grads["dbf"]), ("bi", model.bi, grads["dbi"]),
    ("bg", model.bg, grads["dbg"]), ("bo", model.bo, grads["dbo"]),
    ("W_out", model.W_out, grads["dW_out"]), ("b_out", model.b_out, grads["db_out"]),
    ("X", X, grads["dX"]),
]

print("=== LSTM (BPTT) gradient check: analytic vs numerical (finite-difference) ===\n")
worst = 0.0
for name, param, analytic in checks:
    numeric = numerical_gradient(loss_fn, param)
    diff = np.max(np.abs(analytic - numeric))
    worst = max(worst, diff)
    print(f"{name}: max abs diff = {diff:.2e}")

print("\nPASS" if worst < 1e-4 else "\nFAIL", f"(worst max-abs-diff = {worst:.2e})")
