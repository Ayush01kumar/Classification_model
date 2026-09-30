"""
Hand-worked example: run the TinyCNN forward and backward pass on the
6x6 vertical-edge image in data/hand_example.csv, printing every
intermediate value so the article's worked example is built from
code-verified numbers, not hand arithmetic.
"""
import numpy as np
from cnn_scratch import TinyCNN, conv2d_forward, relu_forward, maxpool_forward

np.set_printoptions(precision=4, suppress=True)

X = np.loadtxt("data/hand_example.csv", delimiter=",")
print("Input image X (6x6):")
print(X)
print()

# 3x3 vertical-edge-detector kernel
K = np.array([[1., 0., -1.],
              [1., 0., -1.],
              [1., 0., -1.]])
b_conv = -1.0

# Small, hand-friendly FC weights: flat (4,) -> logits (2,)
W_fc = np.array([[0.1, 0.5],
                  [0.2, 0.4],
                  [0.1, 0.3],
                  [0.2, 0.4]])
b_fc = np.array([0.1, -0.1])

model = TinyCNN(K, b_conv, W_fc, b_fc)

# --- forward, with intermediate stages shown standalone for clarity ---
Z = conv2d_forward(X, K, b_conv)
print("Conv output Z = X * K + b_conv (4x4):")
print(Z)
print()

A = relu_forward(Z)
print("After ReLU, A (4x4):")
print(A)
print()

P, argmax = maxpool_forward(A, size=2, stride=2)
print("After 2x2 max pooling stride 2, P (2x2):")
print(P)
print("Argmax locations (pooled index -> source index in A):", argmax)
print()

flat = P.flatten()
print("Flattened:", flat)

probs, cache = model.forward(X)
print("Logits:", cache["logits"])
print("Softmax probabilities:", probs)
print()

y_onehot = np.array([0., 1.])  # true class = 1 ("has vertical edge")
loss = -np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))
print("True label (one-hot):", y_onehot)
print("Cross-entropy loss:", loss)
print()

# --- backward pass ---
grads = model.backward(cache, y_onehot)
print("dL/dlogits = p - y:", probs - y_onehot)
print("dL/dW_fc:")
print(grads["dW_fc"])
print("dL/db_fc:", grads["db_fc"])
print("dL/dK (gradient w.r.t. the 3x3 kernel):")
print(grads["dK"])
print("dL/db_conv:", grads["db_conv"])
