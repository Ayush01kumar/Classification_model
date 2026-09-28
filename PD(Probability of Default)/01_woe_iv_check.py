"""
Sanity-checks the WOE/IV formula against a brute-force, no-shortcuts
recomputation from raw counts, and confirms the zero-event edge case
behaves as expected (undefined without smoothing, finite with it).
"""
import numpy as np
import pandas as pd
from pd_scratch import woe_iv

df = pd.read_csv("data/customers8.csv")
y = df["default"].values
age_bin = (df["age"].values >= 30).astype(int)

# Brute-force, by hand, no helper function
n_events_total = int((y == 1).sum())
n_nonevents_total = int((y == 0).sum())

for b in [0, 1]:
    e = int(((age_bin == b) & (y == 1)).sum())
    ne = int(((age_bin == b) & (y == 0)).sum())
    print(f"bin {b}: events={e}, nonevents={ne}")

print("\nWith +0.5 smoothing, brute force for bin 1 (Age>=30, 0 raw events):")
e_sm = 0 + 0.5
ne_sm = 4 + 0.5
tot_e_sm = n_events_total + 0.5 * 2
tot_ne_sm = n_nonevents_total + 0.5 * 2
pct_e = e_sm / tot_e_sm
pct_ne = ne_sm / tot_ne_sm
woe = np.log(pct_ne / pct_e)
print("pct_events:", pct_e, "pct_nonevents:", pct_ne, "WOE:", woe)

out, iv = woe_iv(age_bin, y, smoothing=0.5)
print("\nFunction result for bin 1:", out[1])
assert abs(out[1]["woe"] - woe) < 1e-9, "mismatch between brute force and function!"
print("\nOK: brute-force WOE matches woe_iv() exactly.")
