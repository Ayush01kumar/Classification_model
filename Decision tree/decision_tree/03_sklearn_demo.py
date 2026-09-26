"""scikit-learn Decision Tree demo: classification, pruning path, regression.
Run:  python 03_sklearn_demo.py
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, mean_squared_error, r2_score)

# ---- data: split FIRST, stratified for classification ----
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# ---- an unrestricted tree (overfits) vs a constrained tree ----
full = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
tree = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=5,
                              random_state=42).fit(X_tr, y_tr)

pred  = tree.predict(X_te)              # majority class of the leaf
proba = tree.predict_proba(X_te)[:, 1]  # class-1 fraction in the leaf (Section 15)

print("full  : train %.3f test %.3f leaves %d depth %d" %
      (full.score(X_tr, y_tr), full.score(X_te, y_te), full.get_n_leaves(), full.get_depth()))
print("depth3: train %.3f test %.3f leaves %d" %
      (tree.score(X_tr, y_tr), tree.score(X_te, y_te), tree.get_n_leaves()))
print(confusion_matrix(y_te, pred))
print("precision %.3f recall %.3f F1 %.3f ROC-AUC %.3f" % (
      precision_score(y_te, pred), recall_score(y_te, pred),
      f1_score(y_te, pred), roc_auc_score(y_te, proba)))

# ---- post-pruning: cost-complexity path, alpha chosen by cross-validation on TRAIN only ----
path = full.cost_complexity_pruning_path(X_tr, y_tr)      # effective alphas
search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                      {"ccp_alpha": path.ccp_alphas[:-1]}, cv=5).fit(X_tr, y_tr)
print("best ccp_alpha:", search.best_params_["ccp_alpha"], " test acc:", search.score(X_te, y_te))

# ---- inspect / visualise ----
print(export_text(tree, feature_names=list(load_breast_cancer().feature_names)))
# plot_tree(tree, filled=True)   # needs matplotlib

# ---- regression tree: leaf = mean of y ----
rng = np.random.default_rng(0)
Xr = rng.uniform(0, 10, (200, 1)); yr = np.sin(Xr[:, 0]) + rng.normal(0, 0.2, 200)
reg = DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=0).fit(Xr[:150], yr[:150])
p = reg.predict(Xr[150:])
print("MSE %.3f  R2 %.3f" % (mean_squared_error(yr[150:], p), r2_score(yr[150:], p)))
