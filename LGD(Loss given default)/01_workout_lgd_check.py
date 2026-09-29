"""
Brute-force check of the workout-LGD discounting formula against a
no-shortcuts, by-hand recomputation, for customer 4 (the mortgage loan).
"""
import numpy as np
from lgd_scratch import workout_lgd_r

EAD, t, recovery, cost, r = 500000, 1.5, 550000, 40000, 0.10

# brute force, no helper function
net = recovery - cost
pv = net / (1 + r) ** t
lgd_manual = 1 - pv / EAD
print(f"manual: net={net}, PV={pv:.4f}, LGD={lgd_manual*100:.4f}%")

res = workout_lgd_r(EAD, [(t, recovery, cost)], r)
print(f"function: PV={res['pv_recovery']:.4f}, LGD={res['lgd']*100:.4f}%")

assert abs(res["lgd"] - lgd_manual) < 1e-9, "mismatch between brute force and function!"
print("\nOK: brute-force LGD matches workout_lgd_r() exactly.")

# sanity: r=0 should reduce to the simple undiscounted formula
res0 = workout_lgd_r(EAD, [(t, recovery, cost)], r=0.0)
simple_lgd = 1 - net / EAD
assert abs(res0["lgd"] - simple_lgd) < 1e-9
print(f"sanity check (r=0 reduces to simple LGD): {res0['lgd']*100:.4f}% == {simple_lgd*100:.4f}%  OK")
