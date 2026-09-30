"""
Generate the figures referenced by the RNN Medium article and reference doc.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rnn_scratch import VanillaRNN
import os

os.makedirs("images", exist_ok=True)
np.random.seed(42)

# ---------------------------------------------------------------------
# Figure 1: unrolled-in-time pipeline diagram
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 3))
T = 4
xpos = np.linspace(0.08, 0.92, T)
for i, x in enumerate(xpos):
    ax.text(x, 0.65, f"x_{i+1}", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fce8e6", edgecolor="#cc4125"))
    ax.text(x, 0.35, f"h_{i+1}", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe", edgecolor="#4a86e8"))
    ax.annotate("", xy=(x, 0.46), xytext=(x, 0.57), arrowprops=dict(arrowstyle="->", color="#555"))
    if i < T - 1:
        ax.annotate("", xy=(xpos[i+1]-0.05, 0.35), xytext=(x+0.05, 0.35),
                    arrowprops=dict(arrowstyle="->", color="#555"))
ax.text(xpos[-1] + 0.06, 0.35, "-> dense -> softmax -> loss", ha="left", va="center", fontsize=10)
ax.set_xlim(0, 1.3); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("RNN unrolled through time: same weights (Wxh, Whh, bh) reused at every step")
plt.tight_layout()
plt.savefig("images/rnn_fig1_unrolled.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: hidden state trajectory on the hand example
# ---------------------------------------------------------------------
df = pd.read_csv("data/hand_example.csv")
X = df[["utilization"]].values
Wxh = np.array([[0.5], [0.3]])
Whh = np.array([[0.1, 0.2], [0.4, 0.1]])
bh = np.array([0.0, 0.0])
W_out = np.array([[0.6, -0.3], [0.2, 0.5]])
b_out = np.array([0.0, 0.0])
model = VanillaRNN(Wxh, Whh, bh, W_out, b_out)
probs, cache = model.forward(X)
hs = np.array(cache["hs"])

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(range(1, 5), hs[:, 0], marker="o", label="h_t[0]")
ax.plot(range(1, 5), hs[:, 1], marker="s", label="h_t[1]")
ax.plot(range(1, 5), X.flatten(), marker="^", linestyle="--", color="gray", label="utilization x_t")
ax.set_xlabel("Month (t)"); ax.set_ylabel("Value"); ax.legend()
ax.set_title("Hidden state evolves as utilization rises")
plt.tight_layout()
plt.savefig("images/rnn_fig2_hidden_state.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 3: the vanishing-gradient curve (log scale)
# ---------------------------------------------------------------------
d, n_h, n_classes = 1, 4, 2
Wxh2 = np.random.randn(n_h, d) * 0.5
Whh2 = np.random.randn(n_h, n_h) * 0.5
bh2 = np.zeros(n_h)
W_out2 = np.random.randn(n_h, n_classes) * 0.5
b_out2 = np.zeros(n_classes)
y_onehot = np.array([0., 1.])

Ts = [2, 4, 8, 12, 16, 20, 30]
dh1_norms = []
for T_ in Ts:
    np.random.seed(7)
    Xg = np.random.randn(T_, d) * 0.5
    m = VanillaRNN(Wxh2.copy(), Whh2.copy(), bh2.copy(), W_out2.copy(), b_out2.copy())
    probs_g, cache_g = m.forward(Xg)
    grads_g = m.backward(cache_g, y_onehot)
    dh1_norms.append(grads_g["dh_norms"][0])

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(Ts, dh1_norms, marker="o", color="#cc4125")
ax.set_xlabel("Sequence length T")
ax.set_ylabel("||dh_1|| (gradient at first timestep, log scale)")
ax.set_title("Vanishing gradient: signal reaching the first timestep decays fast")
plt.tight_layout()
plt.savefig("images/rnn_fig3_vanishing_gradient.png", dpi=150)
plt.close()

print("Figures written to images/")
