"""
CNN (Convolutional Neural Network) from scratch.

Implements, with no external deep-learning library required:
  - 2D convolution (cross-correlation, as frameworks actually implement it),
    forward and backward
  - ReLU, forward and backward
  - 2x2 max pooling, forward and backward (gradient routed to the argmax)
  - a dense (fully connected) layer, forward and backward
  - softmax + cross-entropy loss, whose gradient collapses to p - y,
    exactly like the Logistic Regression article's g = p - y, generalized
    from binary to multiclass
  - a tiny end-to-end CNN (conv -> ReLU -> maxpool -> flatten -> dense ->
    softmax) wiring all of the above together
  - a generic numerical-gradient checker, the gold-standard way to verify
    a from-scratch backprop implementation

Everything here is plain NumPy so every number can be reproduced by hand.
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Convolution (cross-correlation), forward and backward
# ---------------------------------------------------------------------------
def conv2d_forward(X, K, b=0.0, stride=1):
    """X: (H, W). K: (kh, kw). Returns Z: (H_out, W_out)."""
    H, W = X.shape
    kh, kw = K.shape
    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1
    Z = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            Z[i, j] = np.sum(patch * K) + b
    return Z


def conv2d_backward(X, K, dZ, stride=1):
    """Returns (dX, dK, db) given the upstream gradient dZ (same shape as Z)."""
    H, W = X.shape
    kh, kw = K.shape
    H_out, W_out = dZ.shape

    dK = np.zeros_like(K)
    dX = np.zeros_like(X)
    for i in range(H_out):
        for j in range(W_out):
            patch = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            dK += patch * dZ[i, j]                                    # dL/dK = correlate(X, dZ)
            dX[i * stride:i * stride + kh, j * stride:j * stride + kw] += K * dZ[i, j]
    db = np.sum(dZ)
    return dX, dK, db


# ---------------------------------------------------------------------------
# 2. ReLU
# ---------------------------------------------------------------------------
def relu_forward(Z):
    return np.maximum(0, Z)


def relu_backward(Z, dA):
    return dA * (Z > 0)


# ---------------------------------------------------------------------------
# 3. Max pooling (2x2, stride 2)
# ---------------------------------------------------------------------------
def maxpool_forward(A, size=2, stride=2):
    H, W = A.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1
    out = np.zeros((H_out, W_out))
    argmax = {}
    for i in range(H_out):
        for j in range(W_out):
            block = A[i * stride:i * stride + size, j * stride:j * stride + size]
            out[i, j] = np.max(block)
            idx = np.unravel_index(np.argmax(block), block.shape)
            argmax[(i, j)] = (i * stride + idx[0], j * stride + idx[1])
    return out, argmax


def maxpool_backward(A_shape, argmax, dOut):
    dA = np.zeros(A_shape)
    for (i, j), (si, sj) in argmax.items():
        dA[si, sj] += dOut[i, j]
    return dA


# ---------------------------------------------------------------------------
# 4. Dense (fully connected) layer
# ---------------------------------------------------------------------------
def dense_forward(x, W, b):
    """x: (n,) flattened input. W: (n, m). b: (m,). Returns y: (m,)."""
    return x @ W + b


def dense_backward(x, W, dy):
    dW = np.outer(x, dy)
    db = dy
    dx = W @ dy
    return dx, dW, db


# ---------------------------------------------------------------------------
# 5. Softmax + cross-entropy
# ---------------------------------------------------------------------------
def softmax(logits):
    z = logits - np.max(logits)     # numerical stability
    e = np.exp(z)
    return e / np.sum(e)


def cross_entropy_loss(p, y_onehot):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(y_onehot * np.log(p))


def softmax_cross_entropy_backward(p, y_onehot):
    """The gradient of cross-entropy w.r.t. the PRE-softmax logits collapses
    to p - y, exactly like g = p - y in the Logistic Regression / PD /
    XGBoost articles, generalized from binary to multiclass."""
    return p - y_onehot


# ---------------------------------------------------------------------------
# 6. Tiny end-to-end CNN: conv -> ReLU -> maxpool -> flatten -> dense -> softmax
# ---------------------------------------------------------------------------
class TinyCNN:
    def __init__(self, K, b_conv, W_fc, b_fc):
        self.K, self.b_conv, self.W_fc, self.b_fc = K, b_conv, W_fc, b_fc

    def forward(self, X):
        Z = conv2d_forward(X, self.K, self.b_conv)
        A = relu_forward(Z)
        P, argmax = maxpool_forward(A, size=2, stride=2)
        flat = P.flatten()
        logits = dense_forward(flat, self.W_fc, self.b_fc)
        probs = softmax(logits)
        cache = dict(X=X, Z=Z, A=A, P=P, argmax=argmax, flat=flat,
                     logits=logits, probs=probs, P_shape=P.shape, A_shape=A.shape)
        return probs, cache

    def backward(self, cache, y_onehot):
        dlogits = softmax_cross_entropy_backward(cache["probs"], y_onehot)
        dflat, dW_fc, db_fc = dense_backward(cache["flat"], self.W_fc, dlogits)
        dP = dflat.reshape(cache["P_shape"])
        dA = maxpool_backward(cache["A_shape"], cache["argmax"], dP)
        dZ = relu_backward(cache["Z"], dA)
        dX, dK, db_conv = conv2d_backward(cache["X"], self.K, dZ)
        grads = dict(dK=dK, db_conv=db_conv, dW_fc=dW_fc, db_fc=db_fc, dX=dX)
        return grads


# ---------------------------------------------------------------------------
# 7. Numerical gradient checker (the gold-standard backprop verification)
# ---------------------------------------------------------------------------
def numerical_gradient(loss_fn, param, eps=1e-5):
    """loss_fn: takes no args, reads `param` by closure, returns scalar loss.
    param: a numpy array, perturbed in place. Returns a same-shaped array of
    numerical (finite-difference) gradients."""
    grad = np.zeros_like(param, dtype=float)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        loss_plus = loss_fn()
        param[idx] = orig - eps
        loss_minus = loss_fn()
        param[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad
