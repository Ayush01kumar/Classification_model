# Random Forest from First Principles

Code and figures for the Medium article "Random Forest from First Principles: Every Number Solved by Hand".

## Files
- `decision_tree_scratch.py` – from-scratch tree (with a hook for choosing candidate features per split)
- `random_forest_scratch.py` – from-scratch forest: bootstrap, random features per split, soft vote, OOB score, MDI importance
- `01_bootstrap_oob_variance.py` – bootstrap probability (1-1/n)^n -> e^-1, OOB share, variance-of-average simulation
- `02_hand_example.py` – the 8-customer, 3-stump pen-and-paper example (votes, OOB 6/8)
- `03_sklearn_demo.py` – scikit-learn classifier and regressor on real data
- `04_verify_against_sklearn.py` – scratch forest vs scikit-learn
- `data/customers8.csv`, `images/` (figures rf_fig1..rf_fig6), `requirements.txt`

## Quick start
```
pip install -r requirements.txt
python 02_hand_example.py
python 04_verify_against_sklearn.py   # ~13 s per 100-tree scratch forest
```

## Math -> code
| Math | Code |
|---|---|
| Bootstrap rows | `rng.integers(0, n, n)` |
| OOB rows | rows not in the bootstrap index |
| m = floor(sqrt(p)) | `_features_to_try` in the randomized tree |
| Gini, gain, midpoint thresholds | `decision_tree_scratch.py` |
| Average of tree probabilities | `predict_proba` |

## Verified results (breast cancer, 75/25 stratified, random_state=42, scikit-learn 1.8.0)
Single tree test 0.923; scikit-learn forest (300 trees) test 0.958, OOB 0.962; scratch forest (100 trees) test 0.958 for seeds 0/1/2 with 100% prediction agreement with scikit-learn; importance correlation ~0.92.

## Caveats
The scratch forest is for learning and is slow. With few trees some rows may have no OOB tree (scikit-learn warns). MDI importance is biased toward high-cardinality features.
