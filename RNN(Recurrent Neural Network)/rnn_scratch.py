"""
Vanilla RNN (Recurrent Neural Network) from scratch.

Implements, with no external deep-learning library required:
  - a single-layer RNN cell (tanh activation), forward and backward
    through time (BPTT, Back-Propagation Through Time)
  - a dense output head reading the final hidden state, + softmax +
    cross-entropy, reusing the same g = p - y result as every other
    article in this series (Logistic Regression, PD, XGBoost, CNN)
  - a generic numerical-gradient checker
  - a vanishing-gradient probe: measures how much gradient signal
    reaches an early timestep as sequence length grows, the motivating
    problem the LSTM article (built separately, deliberately differently)
    solves.

Task used throughout this repo (tying back into the credit-risk theme of
the PD/LGD/EAD articles): classify a short sequence of monthly credit
utilization ratios as "distress trend" (rising toward the limit) vs
"stable", using the SAME 2-class softmax head used in the CNN article.

Everything here is plain NumPy so every number can be reproduced by hand.
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. RNN cell: single timestep, forward and backward
# ---------------------------------------------------------------------------
def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):
    """x_t: (d,). h_prev: (n_h,). Returns h_t: (n_h,) and the pre-activation
    a_t (needed for the tanh backward pass)."""
    a_t = Wxh @ x_t + Whh @ h_prev + bh
    h_t = np.tanh(a_t)
    return h_t, a_t


def rnn_step_backward(dh_t, a_t, x_t, h_prev, Wxh, Whh):
    """Backprop through one RNN cell. dh_t: dL/dh_t (total gradient flowing
    INTO h_t, i.e. from the output head plus from the next timestep).
    Returns dWxh, dWhh, dbh (this step's contribution), dx_t, and
    dh_prev (gradient to propagate one more step back in time)."""
    da_t = dh_t * (1 - np.tanh(a_t) ** 2)          # d tanh(a)/da = 1 - tanh(a)^2
    dWxh = np.outer(da_t, x_t)
    dWhh = np.outer(da_t, h_prev)
    dbh = da_t
    dx_t = Wxh.T @ da_t
    dh_prev = Whh.T @ da_t
    return dWxh, dWhh, dbh, dx_t, dh_prev


# ---------------------------------------------------------------------------
# 2. Dense head + softmax + cross-entropy (identical to the CNN article)
# ---------------------------------------------------------------------------
def dense_forward(x, W, b):
    return x @ W + b


def dense_backward(x, W, dy):
    dW = np.outer(x, dy)
    db = dy
    dx = W @ dy
    return dx, dW, db


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def cross_entropy_loss(p, y_onehot):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(y_onehot * np.log(p))


def softmax_cross_entropy_backward(p, y_onehot):
    """collapses to p - y, exactly like g = p - y throughout this series"""
    return p - y_onehot


# ---------------------------------------------------------------------------
# 3. Full RNN over a sequence: unroll T steps, then classify
# ---------------------------------------------------------------------------
class VanillaRNN:
    def __init__(self, Wxh, Whh, bh, W_out, b_out):
        self.Wxh, self.Whh, self.bh = Wxh, Whh, bh
        self.W_out, self.b_out = W_out, b_out
        self.n_h = Whh.shape[0]

    def forward(self, X):
        """X: (T, d) sequence. Returns probs (n_classes,) and a cache with
        every intermediate needed for BPTT."""
        T = X.shape[0]
        h_prev = np.zeros(self.n_h)
        hs, a_s = [], []
        for t in range(T):
            h_t, a_t = rnn_step_forward(X[t], h_prev, self.Wxh, self.Whh, self.bh)
            hs.append(h_t)
            a_s.append(a_t)
            h_prev = h_t
        logits = dense_forward(hs[-1], self.W_out, self.b_out)
        probs = softmax(logits)
        cache = dict(X=X, hs=hs, a_s=a_s, logits=logits, probs=probs)
        return probs, cache

    def backward(self, cache, y_onehot):
        """Full BPTT: backprop from the loss through the dense head, then
        back through every timestep, accumulating gradients."""
        X, hs, a_s = cache["X"], cache["hs"], cache["a_s"]
        T = X.shape[0]

        dlogits = softmax_cross_entropy_backward(cache["probs"], y_onehot)
        dh_last, dW_out, db_out = dense_backward(hs[-1], self.W_out, dlogits)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dX = np.zeros_like(X)

        dh_next = dh_last  # gradient flowing into the CURRENT timestep's h_t
        dh_norms = []       # ||dh_t|| at each timestep, for the vanishing-gradient probe
        for t in reversed(range(T)):
            h_prev = hs[t - 1] if t > 0 else np.zeros(self.n_h)
            dh_norms.append((t, np.linalg.norm(dh_next)))
            dWxh_t, dWhh_t, dbh_t, dx_t, dh_prev = rnn_step_backward(
                dh_next, a_s[t], X[t], h_prev, self.Wxh, self.Whh)
            dWxh += dWxh_t
            dWhh += dWhh_t
            dbh += dbh_t
            dX[t] = dx_t
            dh_next = dh_prev  # this becomes dh_t for the PREVIOUS timestep

        dh_norms = dict(dh_norms)
        grads = dict(dWxh=dWxh, dWhh=dWhh, dbh=dbh, dW_out=dW_out, db_out=db_out,
                     dX=dX, dh_norms=dh_norms)
        return grads


# ---------------------------------------------------------------------------
# 4. Numerical gradient checker (identical tool to the CNN article)
# ---------------------------------------------------------------------------
def numerical_gradient(loss_fn, param, eps=1e-5):
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
