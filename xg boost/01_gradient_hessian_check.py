"""Numerical checks of the XGBoost formulas (finite differences and brute-force search).
Run: python 01_gradient_hessian_check.py
"""
import numpy as np
from xgboost_scratch import sigmoid

# 1) g = p - y and h = p(1-p) for the logistic loss l(F) = -[y log p + (1-y) log(1-p)], p = sigmoid(F)
def loss(F, y):
    p = sigmoid(F); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

print("1) gradient and Hessian of the logistic loss w.r.t. the margin F")
eps = 1e-4
for y in (0, 1):
    for F in (-2.0, 0.0, 0.7):
        p = sigmoid(F)
        g_num = (loss(F + eps, y) - loss(F - eps, y)) / (2 * eps)
        h_num = (loss(F + eps, y) - 2 * loss(F, y) + loss(F - eps, y)) / eps**2
        print(f"   y={y} F={F:+.1f}  g: formula {p - y:+.6f} numeric {g_num:+.6f} | h: formula {p*(1-p):.6f} numeric {h_num:.6f}")

# 2) optimal leaf weight w* = -G/(H+lambda) minimises  G w + 0.5 (H + lambda) w^2
print("\n2) leaf weight from the second-order objective (round 1 left leaf of the hand example)")
G, H, lam = -1.0, 1.0, 1.0
ws = np.linspace(-3, 3, 600001)
obj = G * ws + 0.5 * (H + lam) * ws**2
print(f"   formula w* = {-G / (H + lam):.4f}   brute-force argmin = {ws[np.argmin(obj)]:.4f}   min objective = {obj.min():.4f}  (= -0.5 G^2/(H+lam) = {-0.5 * G * G / (H + lam):.4f})")

# 3) gain = objective decrease when a leaf is split
print("\n3) split gain equals the drop in the regularised objective (hand example, Age < 30, gamma = 0)")
def leaf_obj(G, H, lam): return -0.5 * G * G / (H + lam)
GL, HL, GR, HR = -1.0, 1.0, 2.0, 1.0
parent = leaf_obj(GL + GR, HL + HR, lam); kids = leaf_obj(GL, HL, lam) + leaf_obj(GR, HR, lam)
gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR) ** 2 / (HL + HR + lam))
print(f"   objective before {parent:.4f}, after {kids:.4f}, drop {parent - kids:.4f}, formula gain {gain:.4f}")

# 4) squared-error mini example used in the reference document
print("\n4) regression by hand: x = [1,2,3,4], y = [1,2,6,7], base = mean(y), lambda = 1, eta = 0.5")
x = np.array([1., 2, 3, 4]); yy = np.array([1., 2, 6, 7]); F = np.full(4, yy.mean())
g, h = F - yy, np.ones(4); Gt, Ht = g.sum(), h.sum()
print("   F0 =", F, " g = F - y =", g, " h =", h, f" G={Gt}, H={Ht}")
for t in (1.5, 2.5, 3.5):
    m = x < t; GL, HL, GR, HR = g[m].sum(), h[m].sum(), g[~m].sum(), h[~m].sum()
    gn = 0.5 * (GL**2 / (HL + 1) + GR**2 / (HR + 1) - Gt**2 / (Ht + 1))
    print(f"   x < {t}: G_L={GL:+.1f} H_L={HL:.0f} G_R={GR:+.1f} H_R={HR:.0f} gain={gn:.4f}")
m = x < 2.5; wl, wr = -g[m].sum() / (m.sum() + 1), -g[~m].sum() / ((~m).sum() + 1)
F1 = F + 0.5 * np.where(m, wl, wr)
print(f"   leaf weights {wl:.4f}, {wr:.4f};  F1 = {F1};  MSE {np.mean((yy-F)**2):.4f} -> {np.mean((yy-F1)**2):.4f}")
