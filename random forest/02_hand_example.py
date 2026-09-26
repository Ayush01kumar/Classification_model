"""The pen-and-paper forest of the article: 8 customers, 3 stumps with fixed bootstrap rows and features.
Run: python 02_hand_example.py
"""
import numpy as np
from decision_tree_scratch import DecisionTreeScratch

data = np.genfromtxt("data/customers8.csv", delimiter=",", skip_header=1)
X, y = data[:, :2], data[:, 2].astype(int)
names = ["Age", "Income"]
samples = [[1, 3, 3, 6, 7, 8, 8, 8], [4, 4, 5, 5, 5, 5, 5, 6], [1, 1, 1, 4, 5, 7, 7, 8]]   # 1-based rows drawn
features = [1, 1, 0]                                                                       # feature searched by each tree
new = np.array([26.0, 32.0])

votes, oob_votes = [], {i: [] for i in range(1, 9)}
for k, (rows, f) in enumerate(zip(samples, features), 1):
    idx = np.array(rows) - 1
    stump = DecisionTreeScratch(max_depth=1).fit(X[idx][:, [f]], y[idx])
    r = stump.root_
    print(f"Tree {k}: rows {rows}  feature {names[f]}  split {names[f]} < {r.threshold:g}  gain {r.gain:.4f}  "
          f"left -> {r.left.value}, right -> {r.right.value}")
    predict = lambda row: stump.predict_one(np.array([row[f]]))
    votes.append(int(predict(new)))
    for i in range(1, 9):
        if i not in rows:
            oob_votes[i].append(int(predict(X[i - 1])))

print("\nVotes for new customer (Age 26, Income 32):", votes, "->", "Risky" if sum(votes) * 2 > len(votes) else "Not risky")
correct = 0
for i in range(1, 9):
    pred = int(sum(oob_votes[i]) * 2 > len(oob_votes[i])); correct += pred == y[i - 1]
    print(f"  row {i}: OOB votes {oob_votes[i]} -> {pred}   actual {y[i - 1]}")
print(f"OOB accuracy = {correct}/8 = {correct / 8:.2f}")
