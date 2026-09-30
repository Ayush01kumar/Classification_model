"""
The vanishing-gradient probe: the central empirical result of this
article. For sequences of increasing length T, run the VanillaRNN
forward and backward, and record the ratio ||dh_1|| / ||dh_T|| -- how
much gradient signal survives the trip from the last timestep back to
the first. With repeated multiplication by Whh^T * diag(1 - tanh(a)^2)
at every step (see rnn_scratch.rnn_step_backward), this ratio shrinks
geometrically, which is exactly why a vanilla RNN struggles to learn
dependencies spanning many timesteps -- the motivating problem the LSTM
article (a deliberately different architecture, not just a renamed RNN)
is built to solve.
"""
import numpy as np
from rnn_scratch import VanillaRNN

np.random.seed(7)
d, n_h, n_classes = 1, 4, 2

Wxh = np.random.randn(n_h, d) * 0.5
Whh = np.random.randn(n_h, n_h) * 0.5
bh = np.zeros(n_h)
W_out = np.random.randn(n_h, n_classes) * 0.5
b_out = np.zeros(n_classes)

y_onehot = np.array([0., 1.])

print("T (sequence length) | ||dh_1|| (gradient at FIRST timestep) | ratio to ||dh_T||")
for T in [2, 4, 8, 12, 16, 20, 30]:
    np.random.seed(7)  # same weights every time, only T changes
    X = np.random.randn(T, d) * 0.5
    model = VanillaRNN(Wxh.copy(), Whh.copy(), bh.copy(), W_out.copy(), b_out.copy())
    probs, cache = model.forward(X)
    grads = model.backward(cache, y_onehot)
    dh1 = grads["dh_norms"][0]
    dhT = grads["dh_norms"][T - 1]
    ratio = dh1 / dhT if dhT > 0 else float("nan")
    print(f"{T:20d} | {dh1:.8f}                      | {ratio:.6f}")

print("\nGradient at the first timestep shrinks by many orders of magnitude as the")
print("sequence gets longer -- the vanishing gradient problem, purely from repeated")
print("multiplication through tanh' (always <= 1) and Whh^T at every backward step.")
