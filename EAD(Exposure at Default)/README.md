# EAD (Exposure at Default) — from scratch

A from-scratch implementation of EAD modeling: CCF (Credit Conversion Factor)
from fixed-horizon historical cohorts, EAD forecasting for revolving credit,
the Basel EAD floor, downturn vs long-run-average CCF, a Beta regression CCF
model, and SA-CCR EAD for derivatives/counterparty credit risk. Companion
code for the Medium article "EAD (Exposure at Default) from First
Principles: Every Number Solved by Hand."

**Note:** `statsmodels` (for its production-grade `BetaModel`) could not be
installed in the environment this was built in (no PyPI access — same
constraint as the PD and LGD articles in this series). `ead_scratch.py`
implements Beta regression from scratch via MLE (scipy.optimize), verified
in `04_verify_beta_regression.py` against `scipy.stats.beta.fit` (exact
match) and a finite-difference gradient check.

## Files

- `ead_scratch.py` — core implementation: `ccf_from_cohort` (realized CCF
  from a historical reference window), `ead_forecast` (EAD for a
  currently-performing account, with the Basel EAD floor), `downturn_ccf`
  (LRA vs downturn), `BetaRegression` (MLE Beta regression for CCF),
  `sa_ccr_ead`/`sa_ccr_pfe` (derivatives EAD via Basel's SA-CCR).
- `data/revolving_accounts3.csv` — 3 defaulted revolving accounts
  (customers 1, 3, 4 from the PD/LGD articles' 8-customer dataset), each
  with limit/drawn balance 12 months before default and at default.
- `01_ccf_check.py` — brute-force verification of the CCF and EAD-forecast
  formulas, plus CCF=0/CCF=1 boundary sanity checks.
- `02_hand_example.py` — the full pen-and-paper example: CCF per account,
  EAD forecast for a new applicant, a term-loan contrast (deterministic
  EAD), downturn CCF, the EAD floor, and a worked SA-CCR example.
- `03_scratch_demo.py` — the CCF pipeline run end-to-end on a 2,000-account
  synthetic revolving portfolio, including EAD forecasts for a fresh batch
  of performing accounts and a downturn scenario.
- `04_verify_beta_regression.py` — validates `BetaRegression` against
  `scipy.stats.beta.fit` and a finite-difference gradient check, plus a
  synthetic CCF model's calibration/discrimination.
- `05_library_note.py` — reference-only: the equivalent `statsmodels`
  `BetaModel` production call (not executed here).
- `06_make_figures.py`, `07_make_table_image.py` — generate all article
  figures into `images/`.

## Quickstart

```bash
pip install numpy scipy pandas matplotlib
python 01_ccf_check.py
python 02_hand_example.py
python 03_scratch_demo.py
python 04_verify_beta_regression.py
python 06_make_figures.py && python 07_make_table_image.py
```

## Key results

- Realized CCF: customer 1 (credit card) = 93.33%, customer 3 = 87.50%,
  customer 4 = 60.00%. Simple average CCF = 80.28%.
- EAD forecast for a new applicant (limit=80,000, drawn=30,000) using the
  average CCF: 70,138.89. Using downturn CCF (85.00%) instead: 72,500.00.
- Term loan contrast: EAD is just the deterministic amortization-schedule
  balance (375,166.51 after 18 months on a 5-year, 10% loan) - no CCF needed.
- EAD floor: a segment with a slightly negative average CCF (-10%) would
  imply EAD below the current drawn balance without the floor; Basel
  requires EAD >= drawn amount.
- SA-CCR: notional=10,000,000, supervisory factor=0.5%, AddOn=PFE=50,000,
  RC=150,000, alpha=1.4 -> EAD=280,000.
- BetaRegression intercept-only fit matches scipy.stats.beta.fit exactly
  (0.0000 difference in fitted mu and phi). CCF model on a 2,000-account
  synthetic portfolio: MAE=0.1276, Spearman rank correlation=0.52.
