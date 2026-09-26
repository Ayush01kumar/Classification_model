"""Scratch random forest vs scikit-learn's RandomForestClassifier (results are statistical, not identical).
Run: python 04_verify_against_sklearn.py     (takes a few tens of seconds)
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from random_forest_scratch import RandomForestScratch

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

for seed in (0, 1, 2):
    mine = RandomForestScratch(n_estimators=100, random_state=seed).fit(X_tr, y_tr)
    ref = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=seed).fit(X_tr, y_tr)
    print(f"seed {seed}: scratch test {np.mean(mine.predict(X_te) == y_te):.3f} OOB {mine.oob_score_:.3f} | "
          f"sklearn test {ref.score(X_te, y_te):.3f} OOB {ref.oob_score_:.3f} | "
          f"prediction agreement {np.mean(mine.predict(X_te) == ref.predict(X_te)):.3f}")

corr = np.corrcoef(mine.feature_importances_, ref.feature_importances_)[0, 1]
print(f"\nCorrelation between scratch and sklearn impurity importances (seed 2): {corr:.3f}")
