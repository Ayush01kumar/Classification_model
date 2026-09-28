"""
End-to-end scratch PD pipeline on a larger synthetic portfolio (2,000
applicants): bin two continuous variables by quantile, compute WOE/IV,
fit the scorecard, scale to points, and report Gini/AUC/KS.
"""
import numpy as np
import pandas as pd
from pd_scratch import woe_iv, LogisticScorecard, points_scaling, to_score, auc_gini_ks

rng = np.random.default_rng(7)
n = 2000
age = rng.integers(20, 65, n)
income = rng.normal(45, 15, n).clip(10, 150)
util = rng.uniform(0, 1, n)          # credit utilization ratio

# ground-truth default probability (used only to draw labels)
logit_true = -1.5 - 0.03 * (age - 40) - 0.02 * (income - 45) + 3.0 * util
p_true = 1 / (1 + np.exp(-logit_true))
default = rng.binomial(1, p_true)

df = pd.DataFrame({"age": age, "income": income, "util": util, "default": default})
print("Default rate:", df["default"].mean().round(3))

# quantile-bin each variable into 4 bins
def qbin(x, q=4):
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    edges[0] -= 1e-6
    return np.digitize(x, edges[1:-1])

age_b = qbin(df["age"].values)
inc_b = qbin(df["income"].values)
util_b = qbin(df["util"].values)

feats = {}
ivs = {}
for name, b in [("age", age_b), ("income", inc_b), ("util", util_b)]:
    woe_map, iv = woe_iv(b, df["default"].values, smoothing=0.5)
    feats[name] = np.array([woe_map[bb]["woe"] for bb in b])
    ivs[name] = iv

print("\nIV by variable:", {k: round(v, 4) for k, v in ivs.items()})

X = np.column_stack([feats["age"], feats["income"], feats["util"]])
model = LogisticScorecard(lr=0.3, n_iter=5000, l2=1.0).fit(X, df["default"].values)
print("\nfitted beta:", model.beta_)

p_hat = model.predict_proba(X)
logit = model.decision_function(X)
score = to_score(logit)
res = auc_gini_ks(df["default"].values, score)
print("\nAUC:", round(res["auc"], 4), "Gini:", round(res["gini"], 4), "KS:", round(res["ks"], 2))
print("Score range:", score.min().round(1), "to", score.max().round(1))
