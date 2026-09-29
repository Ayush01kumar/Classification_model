"""
Brute-force check of the CCF and EAD-forecast formulas against a
no-shortcuts, by-hand recomputation, for customer 1.
"""
from ead_scratch import ccf_from_cohort, ead_forecast

limit_ref, drawn_ref, drawn_default = 50000, 20000, 48000

# brute force, no helper function
undrawn_ref = limit_ref - drawn_ref
ccf_manual = (drawn_default - drawn_ref) / undrawn_ref
print(f"manual: undrawn@ref={undrawn_ref}, CCF={ccf_manual*100:.4f}%")

ccf_fn = ccf_from_cohort(drawn_ref, limit_ref, drawn_default)
print(f"function: CCF={ccf_fn*100:.4f}%")
assert abs(ccf_fn - ccf_manual) < 1e-9, "mismatch!"
print("OK: brute-force CCF matches ccf_from_cohort() exactly.\n")

# EAD forecast, brute force
limit_now, drawn_now, ccf = 80000, 30000, 0.802778
undrawn_now = limit_now - drawn_now
ead_manual = drawn_now + ccf * undrawn_now
print(f"manual EAD: {drawn_now} + {ccf} * {undrawn_now} = {ead_manual:.4f}")

ead_fn = ead_forecast(drawn_now, limit_now, ccf)
print(f"function EAD: {ead_fn:.4f}")
assert abs(ead_fn - ead_manual) < 1e-6, "mismatch!"
print("OK: brute-force EAD matches ead_forecast() exactly.\n")

# sanity: CCF=0 should give EAD = drawn_now (no further drawdown assumed)
assert ead_forecast(drawn_now, limit_now, 0.0) == drawn_now
# sanity: CCF=1 should give EAD = limit_now (full drawdown assumed)
assert ead_forecast(drawn_now, limit_now, 1.0) == limit_now
print("OK: CCF=0 -> EAD=drawn amount; CCF=1 -> EAD=full limit.")
