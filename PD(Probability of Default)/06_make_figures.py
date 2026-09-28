import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pd_scratch import woe_iv, LogisticScorecard, points_scaling, to_score, auc_gini_ks, psi

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
OUT = "images"

# ---------------------------------------------------------------------------
# Fig 1: PD pipeline schematic
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.axis("off")
steps = ["Raw\nvariables", "WOE bin\n+ IV rank", "Logistic\nregression", "Points\nscaling", "Score /\nPD", "Validate:\nGini, KS, PSI"]
xs = np.linspace(0.05, 0.95, len(steps))
for i, (x, s) in enumerate(zip(xs, steps)):
    box = dict(boxstyle="round,pad=0.5", fc="#eaf2ff", ec="#2a5db0", lw=1.5)
    ax.text(x, 0.5, s, ha="center", va="center", fontsize=10.5, bbox=box, transform=ax.transAxes)
    if i < len(steps) - 1:
        ax.annotate("", xy=(xs[i+1]-0.06, 0.5), xytext=(x+0.06, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#2a5db0"))
ax.set_title("PD scorecard pipeline: from raw data to a validated probability of default", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/pd_fig1_pipeline.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 2: WOE bar chart, Age and Income bins (from the 8-row hand example)
# ---------------------------------------------------------------------------
df = pd.read_csv("data/customers8.csv")
y = df["default"].values
age_bin = (df["age"].values >= 30).astype(int)
inc_bin = (df["income_k"].values >= 32.5).astype(int)
age_woe, age_iv = woe_iv(age_bin, y, smoothing=0.5)
inc_woe, inc_iv = woe_iv(inc_bin, y, smoothing=0.0)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
labels_a = ["Age < 30", "Age >= 30"]
vals_a = [age_woe[0]["woe"], age_woe[1]["woe"]]
colors_a = ["#d9534f" if v < 0 else "#3a8f4c" for v in vals_a]
axes[0].bar(labels_a, vals_a, color=colors_a)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title(f"WOE by Age bin  (IV = {age_iv:.3f})")
axes[0].set_ylabel("WOE (smoothed)")
for i, v in enumerate(vals_a):
    axes[0].text(i, v + (0.08 if v >= 0 else -0.18), f"{v:.3f}", ha="center")

labels_i = ["Income < 32.5k", "Income >= 32.5k"]
vals_i = [inc_woe[0]["woe"], inc_woe[1]["woe"]]
colors_i = ["#d9534f" if v < 0 else "#3a8f4c" for v in vals_i]
axes[1].bar(labels_i, vals_i, color=colors_i)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title(f"WOE by Income bin  (IV = {inc_iv:.3f})")
for i, v in enumerate(vals_i):
    axes[1].text(i, v + (0.08 if v >= 0 else -0.18), f"{v:.3f}", ha="center")

plt.suptitle("Negative WOE = riskier bin (over-represented among defaults)")
plt.tight_layout()
plt.savefig(f"{OUT}/pd_fig2_woe.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Build synthetic credit dataset for smooth ROC/KS curves + PSI-over-time demo
# ---------------------------------------------------------------------------
X, yc = make_classification(n_samples=2000, n_features=8, n_informative=5, n_redundant=1,
                             weights=[0.85, 0.15], class_sep=1.2, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, yc, test_size=0.25, stratify=yc, random_state=42)
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
sk = LogisticRegression(max_iter=5000).fit(Xtr_s, ytr)
p_te = sk.predict_proba(Xte_s)[:, 1]
res = auc_gini_ks(yte, p_te)

# ---------------------------------------------------------------------------
# Fig 3: ROC curve with AUC/Gini annotated
# ---------------------------------------------------------------------------
order = np.argsort(-p_te)
y_sorted = yte[order]
n_pos, n_neg = y_sorted.sum(), len(y_sorted) - y_sorted.sum()
tpr = np.concatenate([[0], np.cumsum(y_sorted) / n_pos])
fpr = np.concatenate([[0], np.cumsum(1 - y_sorted) / n_neg])

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot(fpr, tpr, color="#2a5db0", lw=2, label=f"Model (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Random (AUC=0.5)")
ax.fill_between(fpr, tpr, fpr, color="#2a5db0", alpha=0.12)
ax.set_xlabel("False positive rate (good wrongly flagged risky)")
ax.set_ylabel("True positive rate (bad correctly flagged risky)")
ax.set_title(f"ROC curve — Gini = 2×AUC−1 = {res['gini']:.3f}")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/pd_fig3_roc_gini.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 4: KS chart - cumulative good vs bad by decile, max gap marked
# ---------------------------------------------------------------------------
cum_pos = res["cum_pos"]
cum_neg = res["cum_neg"]
x = np.linspace(0, 100, len(cum_pos))
gap = np.abs(cum_pos - cum_neg)
i_max = np.argmax(gap)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(x, cum_pos * 100, color="#d9534f", lw=2, label="Cumulative % bad (defaults) captured")
ax.plot(x, cum_neg * 100, color="#3a8f4c", lw=2, label="Cumulative % good captured")
ax.vlines(x[i_max], cum_neg[i_max] * 100, cum_pos[i_max] * 100, color="black", lw=1.5, ls="--")
ax.text(x[i_max] + 2, (cum_pos[i_max] + cum_neg[i_max]) / 2 * 100,
        f"KS = {res['ks']:.1f}", fontsize=11, fontweight="bold")
ax.set_xlabel("% of population, sorted riskiest (highest score) first")
ax.set_ylabel("Cumulative % captured")
ax.set_title("KS statistic: maximum separation between good and bad")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/pd_fig4_ks.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 5: PSI bar chart (8-row build population vs shifted new-quarter population)
# ---------------------------------------------------------------------------
build_counts = {0: 4, 1: 4}
actual_counts = {0: 2, 1: 6}
psi_val, rows = psi(build_counts, actual_counts)

fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Age < 30", "Age >= 30"]
x_pos = np.arange(2)
width = 0.35
exp_pct = [rows[0][1] * 100, rows[1][1] * 100]
act_pct = [rows[0][2] * 100, rows[1][2] * 100]
ax.bar(x_pos - width/2, exp_pct, width, label="Build population (expected)", color="#2a5db0")
ax.bar(x_pos + width/2, act_pct, width, label="New quarter (actual)", color="#e08a2c")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel("% of population")
ax.set_title(f"Population Stability Index = {psi_val:.3f} (moderate/major shift)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/pd_fig5_psi.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 6: Cumulative (lifetime) PD term structure over 3 years
# ---------------------------------------------------------------------------
cond = [0.02, 0.03, 0.04]
years = [1, 2, 3]
naive_cum = np.cumsum(cond)
survive = np.cumprod([1 - p for p in cond])
true_cum = 1 - survive

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(years, naive_cum * 100, "o--", color="gray", label="Naive sum (wrong)")
ax.plot(years, true_cum * 100, "o-", color="#2a5db0", lw=2, label="Survival-adjusted cumulative PD")
for yv, tv in zip(years, true_cum):
    ax.annotate(f"{tv*100:.2f}%", (yv, tv * 100), textcoords="offset points", xytext=(6, 4))
ax.set_xticks(years)
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative PD (%)")
ax.set_title("Lifetime PD: chaining conditional annual PDs (2%, 3%, 4%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/pd_fig6_term_structure.png", bbox_inches="tight")
plt.close()

print("cumulative PD by year:", true_cum)
print("all figures written to", OUT)
