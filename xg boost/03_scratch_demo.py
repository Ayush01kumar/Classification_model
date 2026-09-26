"""Scratch XGBoost on the breast-cancer data (75/25 stratified split, random_state=42), plus hyperparameter effects.
Run: python 03_scratch_demo.py
"""
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost_scratch import XGBoostScratch, sigmoid

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.25, random_state=42, stratify=data.target)
def logloss(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

t0 = time.time()
base = dict(objective="binary:logistic", n_estimators=100, learning_rate=0.3, max_depth=3, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0)
m = XGBoostScratch(**base).fit(Xtr, ytr)
print(f"scratch XGBoost (100 rounds, depth 3, eta 0.3, lambda 1): test accuracy {np.mean(m.predict(Xte) == yte):.3f}, "
      f"train loss {m.train_loss_[-1]:.4f}, test loss {logloss(yte, m.predict_proba(Xte)[:, 1]):.4f}  ({time.time()-t0:.1f}s)")
print(f"single tree {DecisionTreeClassifier(random_state=42).fit(Xtr, ytr).score(Xte, yte):.3f} | "
      f"random forest {RandomForestClassifier(300, random_state=42).fit(Xtr, ytr).score(Xte, yte):.3f} | "
      f"sklearn GradientBoosting {GradientBoostingClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte):.3f}")

print("\ntrain / test log loss by number of rounds (scratch)")
for r in (0, 1, 5, 10, 25, 50, 100):
    pte = sigmoid(m.decision_function(Xte, n_trees=r))
    print(f"  rounds {r:3d}: train {m.train_loss_[r]:.4f}  test {logloss(yte, pte):.4f}  test acc {np.mean((pte >= .5) == yte):.3f}")

print("\nhyperparameter effects (50 rounds, test log loss / accuracy)")
def run(**kw):
    cfg = dict(base, n_estimators=50); cfg.update(kw)
    mm = XGBoostScratch(**cfg).fit(Xtr, ytr); p = mm.predict_proba(Xte)[:, 1]
    return logloss(yte, p), float(np.mean((p >= .5) == yte))
for label, kw in [("eta 0.3 (base)", {}), ("eta 0.05", {"learning_rate": 0.05}), ("eta 1.0", {"learning_rate": 1.0}),
                  ("lambda 0", {"reg_lambda": 0.0}), ("lambda 10", {"reg_lambda": 10.0}),
                  ("gamma 0", {"gamma": 0.0}), ("gamma 2", {"gamma": 2.0}), ("depth 1", {"max_depth": 1}), ("depth 6", {"max_depth": 6})]:
    ll, acc = run(**kw); print(f"  {label:16s} loss {ll:.4f}  acc {acc:.3f}")

print("\nfeature importance (scratch, share of total gain), top 3")
imp = m.feature_importances_("gain")
for i in np.argsort(imp)[::-1][:3]: print(f"  {data.feature_names[i]:24s} {imp[i]:.3f}")
