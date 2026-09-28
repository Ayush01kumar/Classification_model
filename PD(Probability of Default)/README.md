# PD (Probability of Default) — from scratch

A from-scratch implementation of a credit-risk PD scorecard: WOE/IV binning,
a regularized logistic regression (gradient descent, g = p - y), PDO points
scaling, Gini/AUC/KS, PSI, and lifetime/cumulative PD. Companion code for
the Medium article "Probability of Default (PD) from First Principles:
Every Number Solved by Hand."

**Note:** `scorecardpy` / `optbinning` (the standard production binning
libraries) could not be installed in the environment this was built in
(no PyPI access). `pd_scratch.py` implements the same WOE → IV → logistic
→ points → Gini/KS/PSI chain with plain NumPy, and `04_verify_against_sklearn.py`
confirms the logistic-regression core matches `sklearn.linear_model.LogisticRegression`
to within numerical tolerance (max coefficient diff 0.0008, AUC/KS matching
to 4 decimal places on a 2,000-row synthetic dataset).

## Files

- `pd_scratch.py` — core implementation: `woe_iv`, `LogisticScorecard`,
  `points_scaling`/`to_score`, `auc_gini_ks`, `psi`, `cumulative_pd`.
- `data/customers8.csv` — the 8-customer dataset used across this article
  series (Logistic Regression, Decision Tree, Random Forest, XGBoost, PD).
- `01_woe_iv_check.py` — brute-force verification of the WOE/IV formula,
  including the zero-event (`ln(x/0)`) edge case.
- `02_hand_example.py` — the full pen-and-paper example from the article:
  WOE/IV, unregularized-vs-regularized logistic fit, points scoring,
  Gini/KS, PSI, and the 3-year lifetime PD calculation.
- `03_scratch_demo.py` — the pipeline run end-to-end on a larger (2,000-row)
  synthetic portfolio with quantile-binned continuous variables.
- `04_verify_against_sklearn.py` — validates the scratch logistic regression
  and the Gini/AUC/KS functions against scikit-learn on synthetic data.
- `05_scorecard_library_note.py` — reference-only: the equivalent
  `scorecardpy` production pipeline (not executed here).
- `06_make_figures.py`, `07_make_table_image.py` — generate all article
  figures into `images/`.

## Quickstart

```bash
pip install numpy pandas scikit-learn matplotlib
python 01_woe_iv_check.py
python 02_hand_example.py
python 03_scratch_demo.py
python 04_verify_against_sklearn.py
python 06_make_figures.py && python 07_make_table_image.py
```

## Key results

- WOE(Age<30) = -1.2528, WOE(Age>=30) = +1.7918 (smoothed); IV(Age) = 1.9028, IV(Income) = 0.9704.
- Unregularized logistic regression on WOE features diverges (quasi-complete
  separation on this tiny dataset); L2 (lambda=1) converges to
  beta = [-0.5709, -0.9892, -0.5468].
- Gini = 0.8667, KS = 80.0 on the 8-row example; Gini = 0.7628, KS = 63.84
  on the 2,000-row synthetic verification set (matches sklearn).
- PSI = 0.2747 between the build population and a drifted new-quarter
  population (moderate/major-shift threshold).
- Cumulative 3-year PD from conditional annual PDs 2%/3%/4%: 8.7424%
  (survival-adjusted), vs a naive (wrong) sum of 9.00%.
