# Decision Trees from First Principles

Companion code for the Medium article **"Decision Trees from First Principles: Mathematics, Splitting, and Information Gain"**.

The article derives the algorithm on paper:

```
Data -> candidate split -> class proportions -> impurity (Gini / Entropy) -> weighted child impurity
     -> Information Gain -> best split -> recursion -> stopping / pruning -> leaf prediction
```

Regression uses squared error instead of impurity, and each leaf predicts the mean of its targets.
This repository contains the code behind that mathematics.

## What is inside

| File | Purpose |
|---|---|
| `decision_tree_scratch.py` | Decision tree from scratch in NumPy: class probabilities, Gini, entropy, weighted child impurity, information gain, candidate thresholds, greedy best split, recursive tree building, prediction, class probabilities, and a regression mode (leaf = mean). |
| `01_pen_and_paper_split.py` | Reproduces the article's Age / Risk table: every candidate threshold with child Gini, weights, weighted Gini and gain. Best split: **Age < 26.5, gain 0.25**. |
| `02_customers_tree.py` | Builds the 10-customer tree: **Age < 30 (gain 0.27)**, then **Income < 40 (gain 0.375)**. Repeats it with entropy and predicts three new customers. |
| `03_sklearn_demo.py` | scikit-learn version: train/test split, `DecisionTreeClassifier`, `predict_proba`, metrics, cost-complexity pruning with `ccp_alpha` chosen by cross-validation, tree text export, and `DecisionTreeRegressor`. |
| `04_verify_against_sklearn.py` | Compares the scratch tree with scikit-learn (classification and regression). |
| `data/` | `age_risk.csv` (6 rows) and `customers.csv` (10 rows), the hand-worked examples from the article. |
| `images/` | The nine figures made for the article (anatomy, Gini, entropy, gain flow, split table, recursion, leaf prediction, decision boundaries, overfitting). |

## Quick start

```bash
pip install -r requirements.txt
python 01_pen_and_paper_split.py
python 02_customers_tree.py
python 03_sklearn_demo.py
python 04_verify_against_sklearn.py
```

Run the scripts from inside this folder, because they read `data/` and import `decision_tree_scratch.py` from the current directory.

## Using the scratch tree

```python
import numpy as np
from decision_tree_scratch import DecisionTreeScratch

tree = DecisionTreeScratch(criterion="gini", max_depth=3, min_samples_leaf=1)
tree.fit(X, y)                 # X: (n, p) numeric array, y: integer class labels 0..K-1
tree.predict(X_new)            # majority class of the leaf
tree.predict_proba(X_new)      # class fractions in the leaf
tree.print_tree(names=["Age", "Income"])

reg = DecisionTreeScratch(task="regression", max_depth=3)   # leaf value = mean(y)
```

Parameters: `criterion` (`"gini"` or `"entropy"`), `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_gain`, and `task` (`"classification"` or `"regression"`).

## Mathematics -> code

| Concept | Formula | Function |
|---|---|---|
| Class probability | p_k = n_k / n | `class_probabilities` |
| Gini | G = 1 - sum(p_k^2) | `gini_impurity` |
| Entropy | H = -sum(p_k log2 p_k) | `entropy` |
| Regression impurity | SSE / n | `variance_impurity` |
| Weighted child impurity | I_split = (n_L/n) I_L + (n_R/n) I_R | `weighted_impurity` |
| Information gain | Gain = I_parent - I_split | `information_gain` / `_find_best_split` |
| Candidate thresholds | midpoints of adjacent distinct sorted values | `candidate_thresholds` |
| Best split | argmax of gain over (feature, threshold) | `_find_best_split` |
| Recursion and stopping | BUILD_TREE | `_build_tree` |
| Leaf prediction | argmax p_k, or mean(y) | `_make_leaf` |

## Verification (results of the runs used to write the article)

Run with scikit-learn 1.8.0:

- Age / Risk example: best split Age < 26.5 with gain 0.25 (entropy picks the same threshold, information gain 0.459 bits).
- 10-customer example: scratch tree and scikit-learn give the same predictions.
- XOR data (needs a zero-gain first split): both trees classify it perfectly.
- Breast-cancer dataset, 75/25 stratified split, `max_depth=3, min_samples_leaf=5`: test predictions agree on 100% of rows (accuracy 0.944 for both).
- Breast-cancer dataset, no depth limit: agreement 99.3%. Small differences are expected because exact ties in gain can be broken differently and scikit-learn scans features in a random order (`random_state`).
- Regression, random data, `max_depth=4`: predictions match scikit-learn to floating-point precision.

Your numbers may differ with other library versions or splits.

## Notes and caveats

- This is educational code written for clarity, not speed. Class counts are recomputed at every candidate threshold. scikit-learn sorts once and updates counts incrementally in compiled code.
- Split rule: the scratch tree sends `x < t` left; scikit-learn uses `x <= t`. With midpoint thresholds the partitions are identical.
- `min_gain=0.0` allows zero-gain splits, which is scikit-learn's default behaviour. A positive `min_gain` forbids them.
- The scratch tree has no post-pruning. Cost-complexity pruning (`ccp_alpha`) is shown in the scikit-learn demo.
- Features must be numeric. Categorical and missing-value handling are not implemented in the scratch code, and support in scikit-learn depends on the version.

## Article

Read the full write-up on Medium: [ARTICLE LINK].
