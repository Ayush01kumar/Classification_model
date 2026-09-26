"""Builds the 10-customer tree from the article with the scratch implementation.
Run:  python 02_customers_tree.py
"""
import numpy as np
from decision_tree_scratch import DecisionTreeScratch

data = np.genfromtxt("data/customers.csv", delimiter=",", skip_header=1)
X, y = data[:, :2], data[:, 2].astype(int)
names = ["Age", "Income"]

for criterion in ("gini", "entropy"):
    print(f"--- criterion = {criterion} ---")
    tree = DecisionTreeScratch(criterion=criterion).fit(X, y)
    tree.print_tree(names=names)

tree = DecisionTreeScratch(criterion="gini").fit(X, y)
new = np.array([[26, 32], [52, 40], [23, 45]])
print("\nPredictions for (Age, Income) =", new.tolist())
print("class:", tree.predict(new), " P(risky):", tree.predict_proba(new)[:, 1])
