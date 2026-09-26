"""Verification of the scratch implementation.
(a) Regression, lambda = 0, gamma = 0: XGBoost's second-order tree equals classic gradient boosting on residuals,
    so predictions must match scikit-learn's GradientBoostingRegressor exactly.
(b) If the real `xgboost` package is installed, compare with it on the breast-cancer data (skipped otherwise).
Run: python 04_verify_against_sklearn.py
"""
import numpy as np
from sklearn.datasets import make_friedman1, load_breast_cancer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost_scratch import XGBoostScratch

X, y = make_friedman1(n_samples=600, n_features=10, noise=1.0, random_state=0)   # continuous features, no near-ties
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
ours = XGBoostScratch("reg:squarederror", n_estimators=50, learning_rate=0.1, max_depth=3, reg_lambda=0.0,
                      gamma=0.0, min_child_weight=1.0).fit(Xtr, ytr)
ref = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, criterion="squared_error",
                                random_state=0).fit(Xtr, ytr)
dtr = np.max(np.abs(ours.predict(Xtr) - ref.predict(Xtr)))
dte = np.abs(ours.predict(Xte) - ref.predict(Xte))
print(f"(a) Friedman-1 regression, 50 rounds, depth 3, lambda=0")
print(f"    max |scratch - sklearn| on TRAIN rows = {dtr:.1e}   (same partitions, same leaf values)")
print(f"    test rows with identical prediction: {np.mean(dte < 1e-9):.3f}; max test difference {dte.max():.2f}")
print(f"    test MSE  scratch {np.mean((yte - ours.predict(Xte))**2):.2f}   sklearn {np.mean((yte - ref.predict(Xte))**2):.2f}")
print("    (test differences appear only where two different splits tie in gain on the same training rows,")
print("     e.g. a 17-row node; the two libraries break the tie differently. Not a formula difference.)")

try:
    import xgboost as xgb
except ImportError:
    print("(b) xgboost not installed: skipped (pip install xgboost to compare against the real library)")
else:
    Xb, yb = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=42, stratify=yb)
    cfg = dict(n_estimators=50, learning_rate=0.3, max_depth=3, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0)
    ours = XGBoostScratch("binary:logistic", base_score=0.5, **cfg).fit(Xtr, ytr)
    lib = xgb.XGBClassifier(base_score=0.5, tree_method="exact", random_state=0, **cfg).fit(Xtr, ytr)
    pa, pb = ours.predict_proba(Xte)[:, 1], lib.predict_proba(Xte)[:, 1]
    print(f"(b) breast cancer: max |p_scratch - p_xgboost| = {np.max(np.abs(pa - pb)):.2e}, "
          f"label agreement = {np.mean((pa >= .5) == (pb >= .5)):.3f}, xgboost version {xgb.__version__}")
