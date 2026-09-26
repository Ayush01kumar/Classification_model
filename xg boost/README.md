# XGBoost from First Principles

Code and figures for the Medium article "XGBoost from First Principles: Every Number Solved by Hand".

## Files
- `xgboost_scratch.py` – from-scratch XGBoost: second-order objective, exact greedy splits, lambda / gamma / eta / min_child_weight / max_depth / row and column subsampling, logistic and squared-error objectives, gain / weight / cover importance
- `01_gradient_hessian_check.py` – g = p - y and h = p(1-p) against finite differences; leaf weight and gain against brute force; the regression mini example
- `02_hand_example.py` – the 8-customer, 2-round pen-and-paper example (every number in the article)
- `03_scratch_demo.py` – breast-cancer run and hyperparameter effects
- `04_verify_against_sklearn.py` – (a) lambda=0 regression vs scikit-learn GradientBoostingRegressor; (b) vs the real xgboost package if installed
- `05_xgboost_library_demo.py` – the real `xgboost` library on the same split (**needs `pip install xgboost`; not executed when this repository was built**)
- `data/customers8.csv`, `images/` (xgb_fig1 ... xgb_fig6), `requirements.txt`

## Quick start
```
pip install -r requirements.txt
python 02_hand_example.py
python 03_scratch_demo.py
python 04_verify_against_sklearn.py
```

## Math -> code
| Math | Code |
|---|---|
| g = p - y, h = p(1-p) | `_grad_hess` |
| w = -G/(H+lambda) | `BoostedTree._build` |
| gain = 1/2[...] - gamma | `BoostedTree._best_split` |
| F <- F + eta * w | `XGBoostScratch.fit` |

## Verified results
- Hand example: round-1 gain 1.0833 (Age < 30), weights +0.50 / -1.00, log loss 0.6931 -> 0.6064 -> 0.5429.
- lambda = 0 regression: scratch equals scikit-learn gradient boosting to 3.6e-15 on every training row; 96.7% of test predictions identical (differences are tie-breaks between equal-gain splits).
- Breast cancer (75/25 stratified, random_state=42): scratch XGBoost test accuracy 0.965 (143 test rows; one row = 0.007), test log loss 0.0938 at round 100, minimum 0.0933 at round 75.

## Caveats
The scratch code is for learning: exact splits only, no histogram method, no missing-value default direction, no L1 term, no multiclass, no post-pruning. Library defaults (base_score, tree_method, and so on) vary by xgboost version. The comparison with the real xgboost package was not run where this code was built; run part (b) of `04_verify_against_sklearn.py` and `05_xgboost_library_demo.py` yourself.
