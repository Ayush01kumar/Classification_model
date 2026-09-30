"""
Generate the figures referenced by the LSTM Medium article and reference doc.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lstm_scratch import LSTM
import os

os.makedirs("images", exist_ok=True)
np.random.seed(7)

# ---------------------------------------------------------------------
# Figure 1: LSTM cell diagram (conceptual boxes for the 4 gates + cell state)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
boxes = {
    "x_t, h_prev": (0.08, 0.5),
    "forget f_t": (0.30, 0.85),
    "input i_t": (0.30, 0.62),
    "candidate g_t": (0.30, 0.38),
    "output o_t": (0.30, 0.15),
    "c_t = f*c_prev + i*g": (0.62, 0.5),
    "h_t = o*tanh(c_t)": (0.88, 0.5),
}
colors = {"x_t, h_prev": "#e8eaed", "forget f_t": "#fce8e6", "input i_t": "#e6f4ea",
          "candidate g_t": "#fef7e0", "output o_t": "#e8f0fe",
          "c_t = f*c_prev + i*g": "#d9ead3", "h_t = o*tanh(c_t)": "#cfe2f3"}
for label, (x, y) in boxes.items():
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=colors[label], edgecolor="#555"))
for gate_y in [0.85, 0.62, 0.38, 0.15]:
    ax.annotate("", xy=(0.24, gate_y), xytext=(0.13, 0.5),
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
    ax.annotate("", xy=(0.55, 0.5), xytext=(0.37, gate_y),
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
ax.annotate("", xy=(0.79, 0.5), xytext=(0.72, 0.5), arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("cell-state highway\n(additive, no squashing)", xy=(0.62, 0.68), ha="center", fontsize=8, color="#38761d")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("LSTM cell: three gates control what the additive cell state keeps/writes/exposes")
plt.tight_layout()
plt.savefig("images/lstm_fig1_cell_diagram.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: gate activations + cell/hidden state on the hand example
# ---------------------------------------------------------------------
df = pd.read_csv("data/hand_example.csv")
X = df[["utilization"]].values
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
cs = np.array(cache["cs"]); hs = np.array(cache["hs"])
f_gate = np.array([c["f_t"][0] for c in cache["caches"]])
i_gate = np.array([c["i_t"][0] for c in cache["caches"]])

fig, ax = plt.subplots(figsize=(7, 4.2))
months = range(1, 5)
ax.plot(months, f_gate, marker="o", label="forget gate f_t[0]")
ax.plot(months, i_gate, marker="s", label="input gate i_t[0]")
ax.plot(months, cs[:, 0], marker="^", label="cell state c_t[0]")
ax.plot(months, hs[:, 0], marker="d", label="hidden state h_t[0]")
ax.set_xlabel("Month (t)"); ax.set_ylabel("Value"); ax.legend(fontsize=8)
ax.set_title("Gates, cell state, and hidden state as utilization rises")
plt.tight_layout()
plt.savefig("images/lstm_fig2_gates_states.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 3: gradient retention, LSTM vs RNN, log scale
# ---------------------------------------------------------------------
def rnn_dh1(X, y_onehot, Wxh, Whh, bh, W_out_r, b_out_r):
    T = X.shape[0]; n_h_ = Whh.shape[0]
    h_prev = np.zeros(n_h_); hs_, a_s = [], []
    for t in range(T):
        a_t = Wxh @ X[t] + Whh @ h_prev + bh
        h_t = np.tanh(a_t); hs_.append(h_t); a_s.append(a_t); h_prev = h_t
    logits = hs_[-1] @ W_out_r + b_out_r
    z = logits - np.max(logits); probs_ = np.exp(z) / np.sum(np.exp(z))
    dh_next = W_out_r @ (probs_ - y_onehot)
    dh1 = None
    for t in reversed(range(T)):
        if t == 0:
            dh1 = np.linalg.norm(dh_next)
        da_t = dh_next * (1 - np.tanh(a_s[t]) ** 2)
        dh_next = Whh.T @ da_t
    return dh1

d, n_h2, n_classes = 1, 4, 2
y_onehot = np.array([0., 1.])
Wxh = np.random.randn(n_h2, d) * 0.5
Whh = np.random.randn(n_h2, n_h2) * 0.5
bh = np.zeros(n_h2)
W_out_r = np.random.randn(n_h2, n_classes) * 0.5
b_out_r = np.zeros(n_classes)

Wf2 = np.random.randn(n_h2, n_h2 + d) * 0.5
Wi2 = np.random.randn(n_h2, n_h2 + d) * 0.5
Wg2 = np.random.randn(n_h2, n_h2 + d) * 0.5
Wo2 = np.random.randn(n_h2, n_h2 + d) * 0.5
bf2 = np.ones(n_h2); bi2 = np.zeros(n_h2); bg2 = np.zeros(n_h2); bo2 = np.zeros(n_h2)
W_out_l = np.random.randn(n_h2, n_classes) * 0.5
b_out_l = np.zeros(n_classes)

Ts = [2, 4, 8, 12, 16, 20, 30]
rnn_vals, lstm_vals = [], []
for T_ in Ts:
    np.random.seed(7)
    Xg = np.random.randn(T_, d) * 0.5
    rnn_vals.append(rnn_dh1(Xg, y_onehot, Wxh, Whh, bh, W_out_r, b_out_r))
    m = LSTM(Wf2.copy(), Wi2.copy(), Wg2.copy(), Wo2.copy(), bf2.copy(), bi2.copy(), bg2.copy(), bo2.copy(), W_out_l.copy(), b_out_l.copy())
    p_, c_ = m.forward(Xg)
    g_ = m.backward(c_, y_onehot)
    lstm_vals.append(g_["dh_norms"][0])

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(Ts, rnn_vals, marker="o", color="#cc4125", label="Vanilla RNN")
ax.semilogy(Ts, lstm_vals, marker="s", color="#38761d", label="LSTM")
ax.set_xlabel("Sequence length T")
ax.set_ylabel("||dh_1|| (log scale)")
ax.set_title("LSTM retains far more gradient at long sequence lengths")
ax.legend()
plt.tight_layout()
plt.savefig("images/lstm_fig3_gradient_retention.png", dpi=150)
plt.close()

print("Figures written to images/")
