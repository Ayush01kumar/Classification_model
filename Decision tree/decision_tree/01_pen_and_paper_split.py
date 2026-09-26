"""Reproduces the pen-and-paper split table (Age / Risk example) using the scratch functions.
Run:  python 01_pen_and_paper_split.py
"""
import numpy as np
from decision_tree_scratch import gini_impurity, entropy, candidate_thresholds

data = np.genfromtxt("data/age_risk.csv", delimiter=",", skip_header=1)
age, risk = data[:, 0], data[:, 1].astype(int)      # 1 = Yes (risky)
n = len(risk)

g_parent = gini_impurity(risk, 2)
h_parent = entropy(risk, 2)
print(f"Parent: n={n}, #Yes={risk.sum()}, Gini={g_parent:.3f}, Entropy={h_parent:.3f} bits\n")
print(f"{'t':>6} {'nL':>3} {'yL':>3} {'G_L':>6} {'nR':>3} {'yR':>3} {'G_R':>6} {'wL':>6} {'wR':>6} {'G_split':>8} {'Gain':>6} {'IG(H)':>7}")

best = None
for t in candidate_thresholds(age):
    left, right = risk[age < t], risk[age >= t]
    wl, wr = len(left) / n, len(right) / n
    g_l, g_r = gini_impurity(left, 2), gini_impurity(right, 2)
    g_split = wl * g_l + wr * g_r                     # I_split = (nL/n) I_L + (nR/n) I_R
    gain = g_parent - g_split                          # Gain = I_parent - I_split
    ig = h_parent - (wl * entropy(left, 2) + wr * entropy(right, 2))
    print(f"{t:6.1f} {len(left):3d} {left.sum():3d} {g_l:6.3f} {len(right):3d} {right.sum():3d} {g_r:6.3f} "
          f"{wl:6.3f} {wr:6.3f} {g_split:8.3f} {gain:6.3f} {ig:7.3f}")
    if best is None or gain > best[1]:
        best = (t, gain)

print(f"\nBest split: Age < {best[0]}  (Gini gain = {best[1]:.2f})")
