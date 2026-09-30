"""
LSTM (Long Short-Term Memory) from scratch.

This is deliberately a DIFFERENT architecture from the companion RNN
article's VanillaRNN, not a renamed copy of it: instead of one tanh cell
squashing everything into a single hidden state every step, an LSTM
carries a separate CELL STATE c_t across time, updated ADDITIVELY
(c_t = f_t*c_{t-1} + i_t*g_t) and gated by three learned sigmoid gates.
That additive update is precisely what avoids the repeated-multiplication
shrinkage that caused the vanishing gradient problem in the RNN article
-- verified empirically in 03_gradient_retention_probe.py by re-running
the SAME vanishing-gradient probe used for the RNN and comparing the two.

Implements, with no external deep-learning library required:
  - a single-layer LSTM cell (forget/input/output gates + cell state),
    forward and backward through time (BPTT)
  - the same dense + softmax + cross-entropy head as the RNN/CNN
    articles, so the g = p - y result carries through unchanged
  - a generic numerical-gradient checker
  - a gradient-retention probe, the LSTM counterpart of the RNN
    article's vanishing-gradient probe

Same task as the RNN article (credit-risk themed, for continuity):
classify a short sequence of monthly utilization ratios as a "distress
trend" vs "stable", via a 2-class softmax head reading the final hidden
state.

Everything here is plain NumPy so every number can be reproduced by hand.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. LSTM cell: single timestep, forward and backward
# ---------------------------------------------------------------------------
def lstm_step_forward(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo):
    """x_t: (d,). h_prev, c_prev: (n_h,).
    Concatenated input z = [h_prev, x_t] drives all four gates.
    Returns h_t, c_t, and a cache of every intermediate needed for backward."""
    z = np.concatenate([h_prev, x_t])

    f_t = sigmoid(Wf @ z + bf)   # forget gate: how much of c_prev to keep
    i_t = sigmoid(Wi @ z + bi)   # input gate: how much of the candidate to write
    g_t = np.tanh(Wg @ z + bg)   # candidate cell content
    o_t = sigmoid(Wo @ z + bo)   # output gate: how much of c_t to expose as h_t

    c_t = f_t * c_prev + i_t * g_t     # ADDITIVE update -- the key structural difference vs a plain RNN
    h_t = o_t * np.tanh(c_t)

    cache = dict(z=z, f_t=f_t, i_t=i_t, g_t=g_t, o_t=o_t, c_t=c_t, c_prev=c_prev,
                 tanh_c_t=np.tanh(c_t))
    return h_t, c_t, cache


def lstm_step_backward(dh_t, dc_t_from_future, cache, Wf, Wi, Wg, Wo, n_h):
    """dh_t: dL/dh_t flowing in from the output head / next timestep's input.
    dc_t_from_future: dL/dc_t flowing in directly from timestep t+1's cell
    state (the "cell state highway"). Returns all weight-gradient
    contributions plus dh_prev, dc_prev, and dx_t to keep propagating back."""
    f_t, i_t, g_t, o_t = cache["f_t"], cache["i_t"], cache["g_t"], cache["o_t"]
    c_t, c_prev, tanh_c_t, z = cache["c_t"], cache["c_prev"], cache["tanh_c_t"], cache["z"]

    do_t = dh_t * tanh_c_t
    dc_t = dc_t_from_future + dh_t * o_t * (1 - tanh_c_t ** 2)

    df_t = dc_t * c_prev
    di_t = dc_t * g_t
    dg_t = dc_t * i_t
    dc_prev = dc_t * f_t                      # gradient continuing down the cell-state highway

    da_f = df_t * f_t * (1 - f_t)             # sigmoid' = s*(1-s)
    da_i = di_t * i_t * (1 - i_t)
    da_g = dg_t * (1 - g_t ** 2)               # tanh' = 1 - tanh^2
    da_o = do_t * o_t * (1 - o_t)

    dWf = np.outer(da_f, z); dbf = da_f
    dWi = np.outer(da_i, z); dbi = da_i
    dWg = np.outer(da_g, z); dbg = da_g
    dWo = np.outer(da_o, z); dbo = da_o

    dz = Wf.T @ da_f + Wi.T @ da_i + Wg.T @ da_g + Wo.T @ da_o
    dh_prev = dz[:n_h]
    dx_t = dz[n_h:]

    grads = dict(dWf=dWf, dWi=dWi, dWg=dWg, dWo=dWo,
                 dbf=dbf, dbi=dbi, dbg=dbg, dbo=dbo)
    return grads, dh_prev, dc_prev, dx_t


# ---------------------------------------------------------------------------
# 2. Dense head + softmax + cross-entropy (identical to the RNN/CNN articles)
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
# 3. Full LSTM over a sequence: unroll T steps, then classify
# ---------------------------------------------------------------------------
class LSTM:
    def __init__(self, Wf, Wi, Wg, Wo, bf, bi, bg, bo, W_out, b_out):
        self.Wf, self.Wi, self.Wg, self.Wo = Wf, Wi, Wg, Wo
        self.bf, self.bi, self.bg, self.bo = bf, bi, bg, bo
        self.W_out, self.b_out = W_out, b_out
        self.n_h = bf.shape[0]

    def forward(self, X):
        T = X.shape[0]
        h_prev = np.zeros(self.n_h)
        c_prev = np.zeros(self.n_h)
        hs, cs, caches = [], [], []
        for t in range(T):
            h_t, c_t, cache = lstm_step_forward(
                X[t], h_prev, c_prev, self.Wf, self.Wi, self.Wg, self.Wo,
                self.bf, self.bi, self.bg, self.bo)
            hs.append(h_t); cs.append(c_t); caches.append(cache)
            h_prev, c_prev = h_t, c_t
        logits = dense_forward(hs[-1], self.W_out, self.b_out)
        probs = softmax(logits)
        out_cache = dict(X=X, hs=hs, cs=cs, caches=caches, logits=logits, probs=probs)
        return probs, out_cache

    def backward(self, out_cache, y_onehot):
        X, hs, caches = out_cache["X"], out_cache["hs"], out_cache["caches"]
        T = X.shape[0]

        dlogits = softmax_cross_entropy_backward(out_cache["probs"], y_onehot)
        dh_last, dW_out, db_out = dense_backward(hs[-1], self.W_out, dlogits)

        dWf = np.zeros_like(self.Wf); dWi = np.zeros_like(self.Wi)
        dWg = np.zeros_like(self.Wg); dWo = np.zeros_like(self.Wo)
        dbf = np.zeros_like(self.bf); dbi = np.zeros_like(self.bi)
        dbg = np.zeros_like(self.bg); dbo = np.zeros_like(self.bo)
        dX = np.zeros_like(X)

        dh_next = dh_last
        dc_next = np.zeros(self.n_h)
        dh_norms, dc_norms = [], []
        for t in reversed(range(T)):
            dh_norms.append((t, np.linalg.norm(dh_next)))
            dc_norms.append((t, np.linalg.norm(dc_next)))
            g, dh_prev, dc_prev, dx_t = lstm_step_backward(
                dh_next, dc_next, caches[t], self.Wf, self.Wi, self.Wg, self.Wo, self.n_h)
            dWf += g["dWf"]; dWi += g["dWi"]; dWg += g["dWg"]; dWo += g["dWo"]
            dbf += g["dbf"]; dbi += g["dbi"]; dbg += g["dbg"]; dbo += g["dbo"]
            dX[t] = dx_t
            dh_next, dc_next = dh_prev, dc_prev

        grads = dict(dWf=dWf, dWi=dWi, dWg=dWg, dWo=dWo,
                     dbf=dbf, dbi=dbi, dbg=dbg, dbo=dbo,
                     dW_out=dW_out, db_out=db_out, dX=dX,
                     dh_norms=dict(dh_norms), dc_norms=dict(dc_norms))
        return grads


# ---------------------------------------------------------------------------
# 4. Numerical gradient checker (identical tool to the CNN/RNN articles)
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
