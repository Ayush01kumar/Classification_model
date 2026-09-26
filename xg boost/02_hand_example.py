"""The pen-and-paper XGBoost of the article: 8 customers, 2 boosting rounds of depth-1 trees.
Settings: logistic loss, base probability 0.5 (margin 0), lambda = 1, gamma = 0, eta = 0.3, min_child_weight = 0.
Run: python 02_hand_example.py
"""
import numpy as np
from xgboost_scratch import BoostedTree, sigmoid

data = np.genfromtxt("data/customers8.csv", delimiter=",", skip_header=1)
X, y = data[:, :2], data[:, 2]
names = ["Age", "Income"]
LAM, GAMMA, ETA = 1.0, 0.0, 0.3
F = np.zeros(8)                                    # margins (log-odds); p = sigmoid(F) = 0.5 everywhere

def logloss(y, F):
    p = sigmoid(F); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

print(f"start: p = 0.5 for every row, log loss = {logloss(y, F):.4f}")
for rnd in (1, 2):
    p = sigmoid(F); g, h = p - y, p * (1 - p)
    G, H = g.sum(), h.sum()
    print(f"\n=== Round {rnd} ===")
    print("p =", np.round(p, 4)); print("g = p - y =", np.round(g, 4)); print("h = p(1-p) =", np.round(h, 4))
    print(f"root: G = {G:.4f}, H = {H:.4f}, similarity G^2/(H+lambda) = {G*G/(H+LAM):.4f}")
    print("candidate splits  (x < t goes left)")
    for j, nm in enumerate(names):
        xs = np.unique(X[:, j])
        for t in (xs[:-1] + xs[1:]) / 2:
            m = X[:, j] < t
            GL, HL, GR, HR = g[m].sum(), h[m].sum(), g[~m].sum(), h[~m].sum()
            gain = 0.5 * (GL**2 / (HL + LAM) + GR**2 / (HR + LAM) - G**2 / (H + LAM)) - GAMMA
            print(f"  {nm:6s} < {t:5.1f}  G_L={GL:7.4f} H_L={HL:6.4f}  G_R={GR:7.4f} H_R={HR:6.4f}  gain={gain:.4f}")
    tree = BoostedTree(max_depth=1, reg_lambda=LAM, gamma=GAMMA, min_child_weight=0).fit(X, g, h)
    r = tree.root_
    print(f"best: {names[r.feature]} < {r.threshold:g}, gain = {r.gain:.4f}")
    print(f"  left  leaf: G={r.left.G:.4f} H={r.left.H:.4f} w = -G/(H+lambda) = {r.left.weight:.4f}  -> eta*w = {ETA*r.left.weight:+.4f}")
    print(f"  right leaf: G={r.right.G:.4f} H={r.right.H:.4f} w = -G/(H+lambda) = {r.right.weight:.4f}  -> eta*w = {ETA*r.right.weight:+.4f}")
    F = F + ETA * tree.predict(X)
    print("new margins F =", np.round(F, 4)); print("new p         =", np.round(sigmoid(F), 4))
    print(f"log loss after round {rnd}: {logloss(y, F):.4f}")
    if rnd == 1: tree1 = tree
    else: tree2 = tree

new = np.array([[26.0, 32.0]]); F_new = ETA * (tree1.predict(new) + tree2.predict(new))
print(f"\nNew customer (Age 26, Income 32): margin = {F_new[0]:+.4f}, p(Risky) = {sigmoid(F_new[0]):.4f}")
