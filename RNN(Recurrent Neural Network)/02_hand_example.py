"""
Hand-worked example: run the VanillaRNN forward and backward pass on a
4-month rising-utilization sequence (a "distress trend": the borrower's
credit utilization climbs from 20% to 90% of their limit over 4 months),
printing every intermediate value so the article's worked example is
built from code-verified numbers.
"""
import numpy as np
import pandas as pd
from rnn_scratch import VanillaRNN

np.set_printoptions(precision=4, suppress=True)

df = pd.read_csv("data/hand_example.csv")
X = df[["utilization"]].values  # shape (T, 1)
print("Input sequence X (utilization ratio per month):")
print(X.flatten())
print()

n_h = 2
Wxh = np.array([[0.5], [0.3]])
Whh = np.array([[0.1, 0.2], [0.4, 0.1]])
bh = np.array([0.0, 0.0])
W_out = np.array([[0.6, -0.3], [0.2, 0.5]])
b_out = np.array([0.0, 0.0])

model = VanillaRNN(Wxh, Whh, bh, W_out, b_out)
probs, cache = model.forward(X)

print("Hidden states per timestep (h_t):")
for t, h_t in enumerate(cache["hs"]):
    print(f"  t={t+1}: h_t = {h_t}")
print()

print("Final hidden state h_T:", cache["hs"][-1])
print("Logits:", cache["logits"])
print("Softmax probabilities:", probs)
print()

y_onehot = np.array([0., 1.])  # true class = 1 ("distress trend")
loss = -np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))
print("True label (one-hot):", y_onehot)
print("Cross-entropy loss:", loss)
print()

grads = model.backward(cache, y_onehot)
print("dL/dlogits = p - y:", probs - y_onehot)
print()
print("Gradient norm ||dh_t|| flowing INTO each timestep (BPTT), t=T..1:")
for t in sorted(grads["dh_norms"].keys(), reverse=True):
    print(f"  t={t+1}: ||dh_t|| = {grads['dh_norms'][t]:.6f}")
print()
print("dL/dWxh:\n", grads["dWxh"])
print("dL/dWhh:\n", grads["dWhh"])
print("dL/dbh:", grads["dbh"])
print("dL/dW_out:\n", grads["dW_out"])
print("dL/db_out:", grads["db_out"])
