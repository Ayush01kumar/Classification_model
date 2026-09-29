import numpy as np
import pandas as pd
from lgd_scratch import workout_lgd_r, ltv_haircut_lgd, downturn_lgd

df = pd.read_csv("data/defaulted_loans3.csv")
print(df)

print("\n=== STEP 1: workout LGD per loan, r=10% ===")
results = []
for _, row in df.iterrows():
    cf = [(row["t_years"], row["recovery"], row["cost"])]
    res = workout_lgd_r(row["EAD"], cf, r=0.10)
    results.append(res)
    print(f"customer {row['customer']} ({row['loan_type']}): EAD={row['EAD']}, "
          f"t={row['t_years']}y, recovery={row['recovery']}, cost={row['cost']}")
    print(f"   net cashflow={res['detail'][0]['net']:.2f}, "
          f"PV(net)={res['detail'][0]['pv']:.2f}, LGD={res['lgd']*100:.2f}%")

print("\n=== STEP 2: portfolio (EAD-weighted) LGD ===")
total_ead = df["EAD"].sum()
total_pv = sum(r["pv_recovery"] for r in results)
portfolio_lgd = 1 - total_pv / total_ead
print(f"Total EAD={total_ead}, Total PV recovery={total_pv:.2f}, "
      f"Portfolio LGD={portfolio_lgd*100:.2f}%")

print("\n=== STEP 3: discount rate sensitivity (r=10% vs r=15%) for customer 4 (mortgage) ===")
row4 = df[df.customer == 4].iloc[0]
cf4 = [(row4["t_years"], row4["recovery"], row4["cost"])]
for r in (0.10, 0.15):
    res = workout_lgd_r(row4["EAD"], cf4, r=r)
    print(f"r={r:.0%}: PV={res['pv_recovery']:.2f}, LGD={res['lgd']*100:.2f}%")

print("\n=== STEP 4: LTV/haircut expected recovery, customer 3 (auto) and 4 (mortgage) ===")
row3 = df[df.customer == 3].iloc[0]
for row, haircut in [(row3, 0.30), (row4, 0.15)]:
    res = ltv_haircut_lgd(row["EAD"], row["collateral_value"], haircut)
    print(f"customer {row['customer']}: LTV={res['ltv']*100:.1f}%, haircut={haircut:.0%}, "
          f"recognized collateral={res['recognized_collateral']:.0f}, "
          f"expected recovery={res['expected_recovery']:.0f}, LGD(collateral-only)={res['lgd']*100:.2f}%")

print("\n=== STEP 5: downturn LGD illustration ===")
# simulate 8 "years" of portfolio-average LGD, 2 of which are downturn years
np.random.seed(0)
years_lgd = np.array([0.30, 0.28, 0.32, 0.55, 0.60, 0.29, 0.31, 0.33])  # 2 downturn years (idx 3,4)
downturn_mask = np.array([0, 0, 0, 1, 1, 0, 0, 0])
res = downturn_lgd(years_lgd, downturn_mask)
print(f"LRA LGD (8-year average) = {res['lra_lgd']*100:.2f}%")
print(f"Downturn LGD (2 recession years) = {res['downturn_lgd']*100:.2f}%")
print(f"Regulatory LGD (max of the two) = {res['regulatory_lgd']*100:.2f}%")

print("\n=== STEP 6: stress test customer 3's recovery (used-car market crash, -25%) ===")
stressed_recovery = row3["recovery"] * 0.75
cf3_stress = [(row3["t_years"], stressed_recovery, row3["cost"])]
res_base = workout_lgd_r(row3["EAD"], [(row3["t_years"], row3["recovery"], row3["cost"])], r=0.10)
res_stress = workout_lgd_r(row3["EAD"], cf3_stress, r=0.10)
print(f"base LGD={res_base['lgd']*100:.2f}%, stressed (-25% recovery) LGD={res_stress['lgd']*100:.2f}%")
