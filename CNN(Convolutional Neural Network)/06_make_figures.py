"""
Generate the figures referenced by the Medium article and reference doc.
"""
import numpy as np
import matplotlib.pyplot as plt
from cnn_scratch import (conv2d_forward, relu_forward, maxpool_forward,
                          conv2d_backward, relu_backward, maxpool_backward,
                          dense_forward, dense_backward, softmax,
                          cross_entropy_loss, softmax_cross_entropy_backward)

np.random.seed(42)
import os
os.makedirs("images", exist_ok=True)

# ---------------------------------------------------------------------
# Figure 1: pipeline diagram (conceptual, drawn with matplotlib boxes)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 2.2))
stages = ["Input\nimage", "Conv\n(kernel)", "ReLU", "Max\npool", "Flatten", "Dense", "Softmax", "Loss"]
xpos = np.linspace(0, 1, len(stages))
for x, s in zip(xpos, stages):
    ax.text(x, 0.5, s, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f0fe", edgecolor="#4a86e8"))
for i in range(len(stages) - 1):
    ax.annotate("", xy=(xpos[i+1]-0.03, 0.5), xytext=(xpos[i]+0.03, 0.5),
                arrowprops=dict(arrowstyle="->", color="#555"))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("CNN forward pipeline: conv -> ReLU -> pool -> flatten -> dense -> softmax -> loss")
plt.tight_layout()
plt.savefig("images/cnn_fig1_pipeline.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: hand example, each stage's array as a heatmap
# ---------------------------------------------------------------------
X = np.loadtxt("data/hand_example.csv", delimiter=",")
K = np.array([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
b_conv = -1.0
Z = conv2d_forward(X, K, b_conv)
A = relu_forward(Z)
P, argmax = maxpool_forward(A, size=2, stride=2)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, mat, title in zip(
        axes, [X, Z, A, P],
        ["Input X (6x6)", "Conv output Z (4x4)", "After ReLU, A (4x4)", "After max pool, P (2x2)"]):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_title(title, fontsize=10)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("images/cnn_fig2_hand_example_stages.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 3: kernel as a vertical-edge detector -- visualize K itself
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 3))
im = ax.imshow(K, cmap="RdBu_r", vmin=-1, vmax=1)
for (i, j), v in np.ndenumerate(K):
    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=14)
ax.set_title("Vertical-edge kernel K")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("images/cnn_fig3_kernel.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 4: training curve from 03_scratch_demo.py, rerun compactly here
# ---------------------------------------------------------------------
def make_dataset(n):
    X, y = [], []
    for _ in range(n):
        if np.random.rand() < 0.5:
            edge_col = np.random.choice([2, 3, 4])
            img = np.zeros((6, 6))
            img[:, :edge_col] = 1.0
            img += np.random.randn(6, 6) * 0.15
            X.append(img); y.append(1)
        else:
            img = np.random.rand(6, 6) * 0.9
            X.append(img); y.append(0)
    return np.array(X), np.array(y)

X_train, y_train = make_dataset(200)
Kt = np.random.randn(3, 3) * 0.5
b_conv_t = 0.0
W_fc = np.random.randn(4, 2) * 0.3
b_fc = np.zeros(2)
lr = 0.05
losses = []
for epoch in range(60):
    total_loss = 0.0
    idx = np.random.permutation(len(X_train))
    for i in idx:
        y_onehot = np.zeros(2); y_onehot[y_train[i]] = 1.0
        Z = conv2d_forward(X_train[i], Kt, b_conv_t)
        A = relu_forward(Z)
        Pp, argmax = maxpool_forward(A, size=2, stride=2)
        flat = Pp.flatten()
        logits = dense_forward(flat, W_fc, b_fc)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_onehot)
        total_loss += loss
        dlogits = softmax_cross_entropy_backward(probs, y_onehot)
        dflat, dW_fc, db_fc = dense_backward(flat, W_fc, dlogits)
        dP = dflat.reshape(Pp.shape)
        dA = maxpool_backward(A.shape, argmax, dP)
        dZ = relu_backward(Z, dA)
        _, dK, db_conv_g = conv2d_backward(X_train[i], Kt, dZ)
        Kt -= lr * dK
        b_conv_t -= lr * db_conv_g
        W_fc -= lr * dW_fc
        b_fc -= lr * db_fc
    losses.append(total_loss / len(X_train))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(losses, color="#4a86e8")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean training loss")
ax.set_title("TinyCNN training loss on synthetic edge-detection task")
plt.tight_layout()
plt.savefig("images/cnn_fig4_training_curve.png", dpi=150)
plt.close()

print("Figures written to images/")
