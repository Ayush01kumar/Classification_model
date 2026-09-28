import numpy as np
import pandas as pd
from pd_scratch import (woe_iv, LogisticScorecard, points_scaling, to_score,
                         auc_gini_ks, psi, cumulative_pd)

df = pd.read_csv("data/customers8.csv")
print(df)
y = df["default"].values
age = df["age"].values
inc = df["income_k"].values

print("\n=== STEP 1: bin Age and Income ===")
age_bin = (age >= 30).astype(int)   # 0 = Age<30, 1 = Age>=30
inc_bin = (inc >= 32.5).astype(int) # 0 = Income<32.5, 1 = Income>=32.5
print("age_bin:", age_bin.tolist())
print("inc_bin:", inc_bin.tolist())

print("\n=== STEP 2: WOE/IV for Age, no smoothing (shows the zero-event problem) ===")
try:
    age_woe_raw, age_iv_raw = woe_iv(age_bin, y, smoothing=0.0)
    for b, d in age_woe_raw.items():
        print(b, d)
    print("IV (raw):", age_iv_raw)
except ZeroDivisionError:
    print("Bin Age>=30 has 0 events out of 4 -> %events = 0 -> WOE = ln(x/0) is UNDEFINED.")
    print("This is exactly why practitioners add a smoothing constant. See step 3.")

print("\n=== STEP 3: WOE/IV for Age, with +0.5 smoothing ===")
age_woe, age_iv = woe_iv(age_bin, y, smoothing=0.5)
for b, d in age_woe.items():
    print(b, d)
print("IV (smoothed):", age_iv)

print("\n=== STEP 4: WOE/IV for Income (no smoothing needed) ===")
inc_woe, inc_iv = woe_iv(inc_bin, y, smoothing=0.0)
for b, d in inc_woe.items():
    print(b, d)
print("IV:", inc_iv)

print("\n=== STEP 5: build WOE-transformed feature matrix ===")
X = np.column_stack([
    np.array([age_woe[b]["woe"] for b in age_bin]),
    np.array([inc_woe[b]["woe"] for b in inc_bin]),
])
print("X (WOE_age, WOE_income):\n", X)

print("\n=== STEP 6a: fit UNREGULARIZED logistic regression on WOE features ===")
model_unreg = LogisticScorecard(lr=0.5, n_iter=20000, l2=0.0).fit(X, y)
print("beta unregularized (intercept, b_age, b_income):", model_unreg.beta_)
print("-> note how large these get: this tiny WOE-transformed dataset is (quasi-)")
print("   perfectly separable, so unregularized MLE keeps pushing beta to infinity.")

print("\n=== STEP 6b: fit REGULARIZED logistic regression (l2 = 1.0, same role as XGBoost's lambda) ===")
model = LogisticScorecard(lr=0.5, n_iter=5000, l2=1.0).fit(X, y, verbose_first_step=True)
print("beta (intercept, b_age, b_income):", model.beta_)
fs = model.first_step_
print("\nFirst gradient-descent step (iteration 0):")
print("  p0 (all start at 0.5):", fs["p0"])
print("  g0 = p0 - y:", fs["g0"])
print("  grad0 (d loss / d beta):", fs["grad0"])
beta0_after_1_step = np.array([0.0, 0.0, 0.0]) - 0.5 * fs["grad0"]
print("  beta after 1 step (lr=0.5):", beta0_after_1_step)

print("\n=== STEP 7: fitted probabilities & scores ===")
p_hat = model.predict_proba(X)
logit = model.decision_function(X)
offset, factor = points_scaling(base_score=600, base_odds=50, pdo=20)
print("offset, factor:", offset, factor)
score = to_score(logit, base_score=600, base_odds=50, pdo=20)
out = pd.DataFrame({"age": age, "income_k": inc, "default": y,
                     "WOE_age": X[:, 0], "WOE_income": X[:, 1],
                     "logit": logit, "p_hat": p_hat, "score": score})
print(out.round(4))

print("\n=== STEP 8: Gini / AUC / KS on these 8 rows ===")
res = auc_gini_ks(y, score)
print("AUC:", res["auc"], "Gini:", res["gini"], "KS:", res["ks"])
print("cum_pos (bad captured, sorted riskiest first):", res["cum_pos"])
print("cum_neg (good captured, sorted riskiest first):", res["cum_neg"])

print("\n=== STEP 9: PSI - build population vs a shifted new quarter ===")
# Build (original) age-bin population counts
build_counts = {0: int((age_bin == 0).sum()), 1: int((age_bin == 1).sum())}
# New quarter: 8 new applicants, moderately skewed older
new_ages = np.array([22, 28, 32, 34, 36, 40, 42, 48])
new_age_bin = (new_ages >= 30).astype(int)
actual_counts = {0: int((new_age_bin == 0).sum()), 1: int((new_age_bin == 1).sum())}
print("build_counts (Age bins):", build_counts)
print("actual_counts (Age bins, new quarter):", actual_counts)
psi_val, rows = psi(build_counts, actual_counts)
for r in rows:
    print("bin", r[0], "expected%", round(r[1], 4), "actual%", round(r[2], 4), "contrib", round(r[3], 4))
print("PSI total:", psi_val)

print("\n=== STEP 10: cumulative (lifetime) PD over 3 years ===")
cpd = cumulative_pd([0.02, 0.03, 0.04])
print("naive sum:", 0.02 + 0.03 + 0.04)
print("cumulative PD (survival-adjusted):", cpd)

print("\n=== STEP 11: new customer, Age 26, Income 32 ===")
new_age_b = 1 if 26 >= 30 else 0
new_inc_b = 1 if 32 >= 32.5 else 0
x_new = np.array([[age_woe[new_age_b]["woe"], inc_woe[new_inc_b]["woe"]]])
logit_new = model.decision_function(x_new)[0]
p_new = model.predict_proba(x_new)[0]
score_new = to_score(logit_new, base_score=600, base_odds=50, pdo=20)
print("WOE bins used:", new_age_b, new_inc_b, "logit:", logit_new, "p:", p_new, "score:", score_new)
