"""
End-to-end scratch LGD pipeline on a larger synthetic portfolio (2,000
defaulted loans): LTV + collateral liquidity drive both cure probability
and loss severity; fit the two-stage cure/severity model and report
calibration and discrimination.
"""
import numpy as np
from lgd_scratch import CureSeverityModel, downturn_lgd

rng = np.random.default_rng(11)
n = 2000
ltv = rng.uniform(0.3, 1.6, n)
liquidity = rng.uniform(0, 1, n)          # 1 = very liquid collateral
seniority = rng.integers(0, 2, n)         # 0 = subordinated, 1 = senior
X = np.column_stack([ltv, liquidity, seniority])

p_cure = 1 / (1 + np.exp(-(1.8 - 2.2 * ltv + 1.3 * liquidity + 0.6 * seniority)))
cured = rng.binomial(1, p_cure)

mu_sev = 1 / (1 + np.exp(-(-0.8 + 1.6 * ltv - 1.0 * liquidity - 0.5 * seniority)))
severity = rng.beta(mu_sev * 6, (1 - mu_sev) * 6)
lgd = np.where(cured == 1, rng.uniform(0, 0.005, n), severity)

print("Portfolio cure rate:", round(cured.mean(), 3))
print("Portfolio mean LGD:", round(lgd.mean(), 4))

model = CureSeverityModel(lr=0.3, n_iter=5000, l2=1.0).fit(X, lgd)
lgd_pred = model.predict_lgd(X)

print("\nCure-stage beta (intercept, LTV, liquidity, seniority):", model.cure_beta_.round(3))
print("Severity-stage beta:", model.severity_model_.beta_.round(3),
      " phi:", round(model.severity_model_.phi_, 2))

mae = np.mean(np.abs(lgd_pred - lgd))
corr = np.corrcoef(lgd_pred, lgd)[0, 1]
print(f"\nMAE: {mae:.4f}  Pearson corr: {corr:.4f}")

# downturn illustration: split the portfolio into "years", simulate 2 downturn years
# by scaling up LTV (collateral values fall) and scaling down liquidity
years_lgd = []
for yr in range(8):
    is_downturn = yr in (3, 4)
    ltv_yr = ltv * (1.15 if is_downturn else 1.0)
    liq_yr = liquidity * (0.7 if is_downturn else 1.0)
    Xy = np.column_stack([ltv_yr, liq_yr, seniority])
    years_lgd.append(model.predict_lgd(Xy).mean())

res = downturn_lgd(years_lgd, [0, 0, 0, 1, 1, 0, 0, 0])
print(f"\nSimulated 8-year LGD path: {np.round(years_lgd, 4)}")
print(f"LRA LGD={res['lra_lgd']*100:.2f}%  Downturn LGD={res['downturn_lgd']*100:.2f}%  "
      f"Regulatory LGD={res['regulatory_lgd']*100:.2f}%")
