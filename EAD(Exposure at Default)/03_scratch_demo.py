"""
End-to-end scratch EAD pipeline on a larger synthetic revolving-credit
portfolio (2,000 accounts): utilization, months-to-default and line age
drive CCF; fit the Beta regression CCF model and forecast EAD for a
fresh batch of currently-performing accounts, including a downturn
scenario.
"""
import numpy as np
from ead_scratch import BetaRegression, ead_forecast, downturn_ccf

rng = np.random.default_rng(3)
n = 2000
utilization = rng.uniform(0.1, 0.95, n)
months_to_default = rng.uniform(1, 24, n)
line_age_years = rng.uniform(0.5, 10, n)
X = np.column_stack([utilization, months_to_default, line_age_years])

mu_ccf = 1 / (1 + np.exp(-(0.5 + 1.4 * utilization - 0.03 * months_to_default - 0.05 * line_age_years)))
ccf = rng.beta(mu_ccf * 8, (1 - mu_ccf) * 8)

print("Portfolio mean CCF:", round(ccf.mean(), 4))

model = BetaRegression().fit(X, ccf)
print("Fitted beta (intercept, utilization, months_to_default, line_age):", model.beta_.round(3))
print("Fitted phi:", round(model.phi_, 2))

# forecast EAD for a fresh batch of currently-performing accounts
n_new = 500
limit_new = rng.uniform(20000, 200000, n_new)
util_new = rng.uniform(0.2, 0.9, n_new)
drawn_new = util_new * limit_new
mtd_new = rng.uniform(1, 24, n_new)
age_new = rng.uniform(0.5, 10, n_new)
X_new = np.column_stack([util_new, mtd_new, age_new])
ccf_new = model.predict(X_new)
ead_new = np.array([ead_forecast(d, l, c) for d, l, c in zip(drawn_new, limit_new, ccf_new)])

print(f"\nFresh 500-account batch: mean drawn={drawn_new.mean():.0f}, "
      f"mean forecast EAD={ead_new.mean():.0f}, mean forecast CCF={ccf_new.mean():.4f}")

# downturn scenario: utilization spikes, line age effectively "resets" (new distress)
util_dt = np.clip(util_new * 1.2, 0, 0.99)
X_dt = np.column_stack([util_dt, mtd_new, age_new])
ccf_dt = model.predict(X_dt)
ead_dt = np.array([ead_forecast(d, l, c) for d, l, c in zip(drawn_new, limit_new, ccf_dt)])
print(f"Downturn scenario: mean forecast CCF={ccf_dt.mean():.4f}, mean forecast EAD={ead_dt.mean():.0f} "
      f"(+{(ead_dt.mean() - ead_new.mean()):.0f} vs base case)")
