"""scikit-learn Random Forest: OOB score, importances, permutation importance, effect of max_features.
Run: python 03_sklearn_demo.py
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", oob_score=True,
                            n_jobs=-1, random_state=42).fit(X_tr, y_tr)
print("single tree test accuracy : %.3f" % tree.score(X_te, y_te))
print("random forest test accuracy: %.3f" % rf.score(X_te, y_te))
print("random forest OOB accuracy : %.3f   (no separate validation set needed)" % rf.oob_score_)

proba = rf.predict_proba(X_te)[:5, 1]          # mean of the trees' leaf probabilities
print("first 5 forest probabilities:", np.round(proba, 3))

top = np.argsort(rf.feature_importances_)[::-1][:5]
print("\nTop impurity-based importances (MDI):")
for j in top:
    print("  %-25s %.3f" % (data.feature_names[j], rf.feature_importances_[j]))

perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
top = np.argsort(perm.importances_mean)[::-1][:5]
print("\nTop permutation importances (test set):")
for j in top:
    print("  %-25s %.3f +/- %.3f" % (data.feature_names[j], perm.importances_mean[j], perm.importances_std[j]))

print("\nOOB error vs number of trees and max_features:")
for mf in ("sqrt", None):
    errs = [1 - RandomForestClassifier(n_estimators=n, max_features=mf, oob_score=True, n_jobs=-1,
                                       random_state=0).fit(X, y).oob_score_ for n in (10, 50, 100, 300)]
    print("  max_features=%-5s" % mf, np.round(errs, 4))

# regression forest: prediction = average of the trees' leaf means
rng = np.random.default_rng(0)
Xr = rng.uniform(0, 10, (400, 3)); yr = np.sin(Xr[:, 0]) + 0.3 * Xr[:, 1] + rng.normal(0, 0.2, 400)
reg = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=0, n_jobs=-1).fit(Xr[:300], yr[:300])
pred = reg.predict(Xr[300:])
print("\nRegression forest: test MSE %.3f  R2 %.3f  OOB R2 %.3f" % (
      mean_squared_error(yr[300:], pred), r2_score(yr[300:], pred), reg.oob_score_))
