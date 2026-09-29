import numpy as np
import pandas as pd
from ead_scratch import ccf_from_cohort, ead_forecast, downturn_ccf, sa_ccr_ead, sa_ccr_pfe

df = pd.read_csv("data/revolving_accounts3.csv")
print(df)

print("\n=== STEP 1: realized CCF per historical defaulted account (12mo reference) ===")
ccfs = []
for _, row in df.iterrows():
    ccf = ccf_from_cohort(row["drawn_ref"], row["limit_ref"], row["drawn_default"])
    ccfs.append(ccf)
    undrawn_ref = row["limit_ref"] - row["drawn_ref"]
    print(f"customer {row['customer']}: limit={row['limit_ref']}, drawn@ref={row['drawn_ref']}, "
          f"undrawn@ref={undrawn_ref}, drawn@default={row['drawn_default']}, CCF={ccf*100:.2f}%")

avg_ccf = np.mean(ccfs)
print(f"\nSimple average CCF across 3 accounts = {avg_ccf*100:.2f}%")

print("\n=== STEP 2: forecast EAD for a new, currently-performing account ===")
# New customer, Age 26, Income 32 (continuing this series' running example)
limit_now, drawn_now = 80000, 30000
ead = ead_forecast(drawn_now, limit_now, avg_ccf)
print(f"New applicant (Age 26, Income 32): limit={limit_now}, drawn today={drawn_now}, "
      f"undrawn={limit_now - drawn_now}")
print(f"Forecast EAD = {drawn_now} + {avg_ccf:.4f} * {limit_now - drawn_now} = {ead:.2f}")

print("\n=== STEP 3: contrast with a term loan (deterministic EAD) ===")
# a simple amortizing loan: EAD is just the scheduled outstanding balance
principal, annual_rate, years, months_elapsed = 500000, 0.10, 5, 18
n_months = years * 12
r_m = annual_rate / 12
emi = principal * r_m * (1 + r_m) ** n_months / ((1 + r_m) ** n_months - 1)
balance = principal
for _ in range(months_elapsed):
    interest = balance * r_m
    principal_paid = emi - interest
    balance -= principal_paid
print(f"Term loan: principal={principal}, EMI={emi:.2f}, outstanding after {months_elapsed} months "
      f"= EAD = {balance:.2f} (no CCF needed - it's just the amortization schedule)")

print("\n=== STEP 4: downturn CCF ===")
years_ccf = np.array([0.55, 0.50, 0.58, 0.82, 0.88, 0.52, 0.56, 0.60])
downturn_mask = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
res = downturn_ccf(years_ccf, downturn_mask)
print(f"LRA CCF = {res['lra_ccf']*100:.2f}%, Downturn CCF = {res['downturn_ccf']*100:.2f}%, "
      f"Regulatory CCF = {res['regulatory_ccf']*100:.2f}%")

ead_downturn = ead_forecast(drawn_now, limit_now, res['regulatory_ccf'])
print(f"EAD for the same new applicant using downturn CCF instead: {ead_downturn:.2f} "
      f"(vs {ead:.2f} using the long-run average)")

print("\n=== STEP 5: EAD floor illustration ===")
# an account that has actually PAID DOWN since the reference date - naive CCF formula
# could imply EAD < current drawn balance, which Basel's floor forbids
drawn_now2, limit_now2, ccf_bad = 45000, 50000, -0.10  # a segment with slightly negative avg CCF
ead_unfloor = drawn_now2 + ccf_bad * (limit_now2 - drawn_now2)
ead_floored = ead_forecast(drawn_now2, limit_now2, ccf_bad, floor_at_drawn=True)
print(f"drawn={drawn_now2}, limit={limit_now2}, CCF={ccf_bad}: "
      f"unfloored EAD={ead_unfloor:.2f}, floored EAD={ead_floored:.2f} (Basel floor = drawn amount)")

print("\n=== STEP 6: SA-CCR EAD for a derivative exposure ===")
notional = 10_000_000
supervisory_factor = 0.005   # e.g. an interest rate swap bucket
pfe_res = sa_ccr_pfe(notional, supervisory_factor, maturity_factor=1.0, multiplier=1.0)
replacement_cost = 150000    # current mark-to-market exposure
ead_res = sa_ccr_ead(replacement_cost, pfe_res["pfe"], alpha=1.4)
print(f"Notional={notional}, supervisory factor={supervisory_factor}, AddOn={pfe_res['add_on']:.2f}, "
      f"PFE={pfe_res['pfe']:.2f}")
print(f"RC={ead_res['RC']}, alpha={ead_res['alpha']}, SA-CCR EAD = "
      f"{ead_res['alpha']} * ({ead_res['RC']} + {ead_res['PFE']:.2f}) = {ead_res['EAD']:.2f}")
