"""Compares the scratch tree with scikit-learn on several datasets.
Exact ties in split gain can be broken differently, so agreement is reported as a rate.
Run:  python 04_verify_against_sklearn.py
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from decision_tree_scratch import DecisionTreeScratch

def agree(a, b):
    return float(np.mean(a == b))

# 1) the article's 10-customer example
data = np.genfromtxt("data/customers.csv", delimiter=",", skip_header=1)
X, y = data[:, :2], data[:, 2].astype(int)
mine = DecisionTreeScratch().fit(X, y).predict(X)
sk = DecisionTreeClassifier(random_state=0).fit(X, y).predict(X)
print("customers (train rows) agreement:", agree(mine, sk))

# 2) XOR: needs zero-gain splits (scikit-learn default allows them; so does the scratch code)
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); yx = np.array([0, 1, 1, 0])
print("XOR scratch:", DecisionTreeScratch().fit(Xx, yx).predict(Xx),
      " sklearn:", DecisionTreeClassifier(random_state=0).fit(Xx, yx).predict(Xx))

# 3) breast cancer
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
for label, kw in (("depth<=3, min_leaf=5", dict(max_depth=3, min_samples_leaf=5)), ("unrestricted", dict())):
    s = DecisionTreeScratch(**kw).fit(X_tr, y_tr)
    k = DecisionTreeClassifier(random_state=42, **kw).fit(X_tr, y_tr)
    print(f"breast cancer [{label}] agreement={agree(s.predict(X_te), k.predict(X_te)):.3f} "
          f"scratch acc={agree(s.predict(X_te), y_te):.3f} sklearn acc={k.score(X_te, y_te):.3f}")

# 4) regression
rng = np.random.default_rng(0)
Xr = rng.uniform(0, 10, (300, 2)); yr = np.sin(Xr[:, 0]) + 0.3 * Xr[:, 1] + rng.normal(0, 0.2, 300)
s = DecisionTreeScratch(task="regression", max_depth=4).fit(Xr[:200], yr[:200])
k = DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xr[:200], yr[:200])
print("regression max |scratch - sklearn| on held-out rows:",
      float(np.max(np.abs(s.predict(Xr[200:]) - k.predict(Xr[200:])))))
