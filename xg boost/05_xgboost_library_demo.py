"""The real `xgboost` library on the same breast-cancer split.
NOTE: this script needs `pip install xgboost`. It was NOT executed in the environment where the rest of this
repository was built (the package could not be installed there), so run it yourself and compare with 03_scratch_demo.py.
Run: python 05_xgboost_library_demo.py
"""
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.25, random_state=42, stratify=data.target)

model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.3, max_depth=3,    # rounds, eta, tree depth
    reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,     # lambda, gamma, minimum Hessian sum per child
    subsample=1.0, colsample_bytree=1.0,
    eval_metric="logloss", random_state=42)
model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
print("test accuracy:", round(float(np.mean(model.predict(Xte) == yte)), 3))

hist = model.evals_result()["validation_0"]["logloss"]
print("test log loss: round 1 %.4f, round 50 %.4f, round 100 %.4f" % (hist[0], hist[49], hist[-1]))

imp = model.get_booster().get_score(importance_type="gain")        # average gain per split, by feature name f0, f1, ...
top = sorted(imp.items(), key=lambda kv: -kv[1])[:3]
print("top features by gain:", [(data.feature_names[int(k[1:])], round(v, 2)) for k, v in top])
