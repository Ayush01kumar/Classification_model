"""
Verify TinyCNN.backward() against the numerical (finite-difference)
gradient checker, for every parameter (K, b_conv, W_fc, b_fc) and also
the input X. This is the CNN-appropriate substitute for the
scipy/sklearn cross-checks used in the PD/LGD/EAD articles, since no
deep-learning library is installable in this environment.

NOTE ON THE HAND EXAMPLE: the article's 6x6 hand-worked image is
deliberately symmetric (every row identical), which makes the ReLU/
max-pool feature map have exact TIES between elements. At an exact tie,
the loss surface has a kink -- the one-sided finite-difference estimate
can route through a different argmax than the analytic backward pass,
so dX (specifically) will NOT match at a tied input. This is expected,
correct behavior at a non-differentiable point, not a bug (dK, db_conv,
dW_fc, db_fc are unaffected by this, since the argmax/ReLU mask is the
same object at each layer regardless of which tied pixel "wins"). To
give a clean, unambiguous check of every gradient including dX, this
script instead uses a random, non-symmetric input, where no such ties
exist.
"""
import numpy as np
from cnn_scratch import TinyCNN, numerical_gradient

np.random.seed(0)
np.set_printoptions(precision=6, suppress=True)

X = np.round(np.random.randn(6, 6), 3)
K = np.array([[1., 0., -1.], [0.5, -0.2, -1.], [1., 0.3, -1.]])
b_conv = -0.3
W_fc = np.array([[0.1, 0.5], [0.2, 0.4], [0.1, 0.3], [0.2, 0.4]])
b_fc = np.array([0.1, -0.1])
y_onehot = np.array([0., 1.])

model = TinyCNN(K.copy(), b_conv, W_fc.copy(), b_fc.copy())


def loss_fn():
    probs, _ = model.forward(X)
    return -np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))


probs, cache = model.forward(X)
grads = model.backward(cache, y_onehot)

print("=== Gradient check: analytic (backprop) vs numerical (finite-difference) ===\n")

# K
num_dK = numerical_gradient(loss_fn, model.K)
print("dK analytic:\n", grads["dK"])
print("dK numerical:\n", num_dK)
print("max abs diff:", np.max(np.abs(grads["dK"] - num_dK)))
print()

# b_conv (0-d numeric check via a 1-element array wrapper)
b_conv_arr = np.array([model.b_conv])
def loss_fn_bconv():
    model.b_conv = b_conv_arr[0]
    probs, _ = model.forward(X)
    return -np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))
num_db_conv = numerical_gradient(loss_fn_bconv, b_conv_arr)
model.b_conv = b_conv_arr[0]
print("db_conv analytic:", grads["db_conv"])
print("db_conv numerical:", num_db_conv[0])
print("abs diff:", abs(grads["db_conv"] - num_db_conv[0]))
print()

# W_fc
num_dW_fc = numerical_gradient(loss_fn, model.W_fc)
print("dW_fc analytic:\n", grads["dW_fc"])
print("dW_fc numerical:\n", num_dW_fc)
print("max abs diff:", np.max(np.abs(grads["dW_fc"] - num_dW_fc)))
print()

# b_fc
num_db_fc = numerical_gradient(loss_fn, model.b_fc)
print("db_fc analytic:", grads["db_fc"])
print("db_fc numerical:", num_db_fc)
print("max abs diff:", np.max(np.abs(grads["db_fc"] - num_db_fc)))
print()

# X (dL/dX, less commonly used but a full check of conv2d_backward's dX path)
num_dX = numerical_gradient(loss_fn, X)
print("dX analytic:\n", grads["dX"])
print("dX numerical:\n", num_dX)
print("max abs diff:", np.max(np.abs(grads["dX"] - num_dX)))

print("\nAll gradients match to finite-difference precision (~1e-5 to 1e-7):",
      "PASS" if max(
          np.max(np.abs(grads["dK"] - num_dK)),
          abs(grads["db_conv"] - num_db_conv[0]),
          np.max(np.abs(grads["dW_fc"] - num_dW_fc)),
          np.max(np.abs(grads["db_fc"] - num_db_fc)),
          np.max(np.abs(grads["dX"] - num_dX)),
      ) < 1e-4 else "FAIL")
