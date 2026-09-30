"""
Verify VanillaRNN.backward() (full BPTT) against the numerical
(finite-difference) gradient checker, for every parameter (Wxh, Whh, bh,
W_out, b_out) and the input X, on a random non-degenerate sequence.
"""
import numpy as np
from rnn_scratch import VanillaRNN, numerical_gradient

np.random.seed(0)
np.set_printoptions(precision=6, suppress=True)

T, d, n_h, n_classes = 5, 3, 4, 2
X = np.round(np.random.randn(T, d) * 0.5, 3)
Wxh = np.random.randn(n_h, d) * 0.3
Whh = np.random.randn(n_h, n_h) * 0.3
bh = np.random.randn(n_h) * 0.1
W_out = np.random.randn(n_h, n_classes) * 0.3
b_out = np.random.randn(n_classes) * 0.1
y_onehot = np.array([0., 1.])

model = VanillaRNN(Wxh.copy(), Whh.copy(), bh.copy(), W_out.copy(), b_out.copy())


def loss_fn():
    probs, _ = model.forward(X)
    return -np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))


probs, cache = model.forward(X)
grads = model.backward(cache, y_onehot)

checks = [
    ("Wxh", model.Wxh, grads["dWxh"]),
    ("Whh", model.Whh, grads["dWhh"]),
    ("bh", model.bh, grads["dbh"]),
    ("W_out", model.W_out, grads["dW_out"]),
    ("b_out", model.b_out, grads["db_out"]),
    ("X", X, grads["dX"]),
]

print("=== RNN (BPTT) gradient check: analytic vs numerical (finite-difference) ===\n")
worst = 0.0
for name, param, analytic in checks:
    numeric = numerical_gradient(loss_fn, param)
    diff = np.max(np.abs(analytic - numeric))
    worst = max(worst, diff)
    print(f"{name}: max abs diff = {diff:.2e}")

print("\nPASS" if worst < 1e-4 else "\nFAIL", f"(worst max-abs-diff = {worst:.2e})")
