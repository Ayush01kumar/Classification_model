# LGD (Loss Given Default) — from scratch

A from-scratch implementation of LGD modeling: discounted workout LGD,
LTV/haircut-based collateral recovery, downturn vs long-run-average LGD,
a Beta regression for continuous bounded LGD, and a two-stage cure/severity
model. Companion code for the Medium article "Loss Given Default (LGD)
from First Principles: Every Number Solved by Hand."

**Note:** `statsmodels` (for its production-grade `BetaModel`) could not
be installed in the environment this was built in (no PyPI access).
`lgd_scratch.py` implements Beta regression from scratch via MLE
(scipy.optimize), and `04_verify_beta_regression.py` verifies it two ways:
an intercept-only fit matches `scipy.stats.beta.fit` exactly, and a
finite-difference gradient check confirms the hand-derived log-likelihood.

## Files

- `lgd_scratch.py` — core implementation: `workout_lgd_r` (discounted
  recovery LGD), `ltv_haircut_lgd` (collateral-based recovery), `downturn_lgd`
  (LRA vs downturn), `BetaRegression` (MLE Beta regression), `CureSeverityModel`
  (two-stage logistic-cure + Beta-severity model).
- `data/defaulted_loans3.csv` — 3 defaulted loans (customers 1, 3, 4 from the
  PD article's 8-customer dataset), each with EAD, collateral, recovery
  timing and costs.
- `01_workout_lgd_check.py` — brute-force verification of the discounting
  formula, including the r=0 sanity check.
- `02_hand_example.py` — the full pen-and-paper example: workout LGD per
  loan, portfolio LGD, discount-rate sensitivity, LTV/haircut recovery,
  downturn LGD, and a recovery stress test.
- `03_scratch_demo.py` — the cure/severity pipeline run end-to-end on a
  2,000-loan synthetic portfolio, including a simulated downturn LGD path.
- `04_verify_beta_regression.py` — validates `BetaRegression` against
  `scipy.stats.beta.fit` and a finite-difference gradient check, plus the
  two-stage model's calibration/discrimination on synthetic data.
- `05_library_note.py` — reference-only: the equivalent `statsmodels`
  `BetaModel` production call (not executed here).
- `06_make_figures.py`, `07_make_table_image.py` — generate all article
  figures into `images/`.

## Quickstart

```bash
pip install numpy scipy pandas matplotlib
python 01_workout_lgd_check.py
python 02_hand_example.py
python 03_scratch_demo.py
python 04_verify_beta_regression.py
python 06_make_figures.py && python 07_make_table_image.py
```

## Key results

- Workout LGD (r=10%): customer 1 (unsecured) = 81.82%, customer 3 (auto,
  4-month workout) = 46.72%, customer 4 (mortgage, 18-month workout) = 11.59%.
  EAD-weighted portfolio LGD = 31.10%.
- Discount-rate sensitivity: customer 4's LGD moves from 11.59% (r=10%) to
  17.29% (r=15%) on identical nominal recoveries.
- LTV/haircut recovery: customer 3 (LTV=120%, 30% haircut) implies
  LGD=41.67% from collateral alone; customer 4 (LTV=83%, 15% haircut) is
  fully covered even after the haircut (LGD=0%).
- Downturn illustration: LRA LGD=37.25% vs downturn-years LGD=57.50% on
  an 8-year simulated path -> regulatory LGD = 57.50% (the max of the two).
- BetaRegression intercept-only fit matches scipy.stats.beta.fit exactly
  (0.0000 difference in fitted mu and phi).
- Two-stage cure/severity model on a 2,000-loan synthetic portfolio: MAE
  0.1985-0.2245 depending on the run, materially better than a naive
  mean-only baseline (~0.27-0.28 MAE).
