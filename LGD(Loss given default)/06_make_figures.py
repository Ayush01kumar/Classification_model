import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lgd_scratch import workout_lgd_r, ltv_haircut_lgd, downturn_lgd, CureSeverityModel

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
OUT = "images"

# ---------------------------------------------------------------------------
# Fig 1: LGD pipeline schematic
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.axis("off")
steps = ["Default\noccurs", "Recovery\ncash flows", "Discount\n+ costs", "Workout\nLGD", "Downturn\nadjust", "Model &\nvalidate"]
xs = np.linspace(0.05, 0.95, len(steps))
for i, (x, s) in enumerate(zip(xs, steps)):
    box = dict(boxstyle="round,pad=0.5", fc="#fdeee6", ec="#c0562a", lw=1.5)
    ax.text(x, 0.5, s, ha="center", va="center", fontsize=10.5, bbox=box, transform=ax.transAxes)
    if i < len(steps) - 1:
        ax.annotate("", xy=(xs[i+1]-0.06, 0.5), xytext=(x+0.06, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#c0562a"))
ax.set_title("LGD pipeline: default to a validated, downturn-adjusted loss estimate", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/lgd_fig1_pipeline.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 2: discounting waterfall for the 3 hand-example loans
# ---------------------------------------------------------------------------
df = pd.read_csv("data/defaulted_loans3.csv")
labels, nominal_net, pv_net, ead = [], [], [], []
for _, row in df.iterrows():
    res = workout_lgd_r(row["EAD"], [(row["t_years"], row["recovery"], row["cost"])], r=0.10)
    labels.append(f"Cust {row['customer']}\n({row['loan_type'].replace('_',' ')})")
    nominal_net.append(row["recovery"] - row["cost"])
    pv_net.append(res["pv_recovery"])
    ead.append(row["EAD"])

x = np.arange(3)
width = 0.25
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - width, ead, width, label="EAD", color="#8a8a8a")
ax.bar(x, nominal_net, width, label="Nominal net recovery", color="#e08a2c")
ax.bar(x + width, pv_net, width, label="Discounted (PV) recovery", color="#2a5db0")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("₹")
ax.set_title("EAD vs nominal vs discounted recovery — the gap is loss (LGD)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/lgd_fig2_discounting.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 3: LGD vs LTV curve (haircut model)
# ---------------------------------------------------------------------------
ltv_range = np.linspace(0.3, 1.6, 100)
EAD_fixed = 100.0
haircut = 0.20
lgds = []
for ltv in ltv_range:
    collateral = EAD_fixed / ltv
    res = ltv_haircut_lgd(EAD_fixed, collateral, haircut)
    lgds.append(res["lgd"])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(ltv_range * 100, np.array(lgds) * 100, color="#c0562a", lw=2.2)
ax.axvline(100, color="gray", ls="--", lw=1)
ax.text(101, 5, "LTV=100%", rotation=90, va="bottom", fontsize=9, color="gray")
ax.set_xlabel("Loan-to-Value at default (%)")
ax.set_ylabel("LGD (%), 20% collateral haircut")
ax.set_title("LGD rises sharply once LTV crosses ~100% (haircut-adjusted)")
plt.tight_layout()
plt.savefig(f"{OUT}/lgd_fig3_ltv_curve.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 4: downturn vs LRA LGD
# ---------------------------------------------------------------------------
years_lgd = np.array([0.30, 0.28, 0.32, 0.55, 0.60, 0.29, 0.31, 0.33])
downturn_mask = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
res = downturn_lgd(years_lgd, downturn_mask)
years = [f"Y{i+1}" for i in range(8)]
colors = ["#d9534f" if m else "#2a5db0" for m in downturn_mask]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(years, years_lgd * 100, color=colors)
ax.axhline(res["lra_lgd"] * 100, color="#2a5db0", ls="--", lw=1.5,
           label=f"LRA LGD = {res['lra_lgd']*100:.1f}%")
ax.axhline(res["downturn_lgd"] * 100, color="#d9534f", ls="--", lw=1.5,
           label=f"Downturn LGD = {res['downturn_lgd']*100:.1f}%")
ax.set_ylabel("Portfolio LGD (%)")
ax.set_title("Downturn years (red) push LGD well above the long-run average")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/lgd_fig4_downturn.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 5: bimodal LGD distribution (cure spike + severity spread)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n3 = 2000
ltv = rng.uniform(0.4, 1.4, n3)
liquidity = rng.uniform(0, 1, n3)
p_cure = 1 / (1 + np.exp(-(2.0 - 2.5 * ltv + 1.5 * liquidity)))
cured = rng.binomial(1, p_cure)
mu_sev = 1 / (1 + np.exp(-(-1.0 + 1.8 * ltv - 1.2 * liquidity)))
severity = rng.beta(mu_sev * 6, (1 - mu_sev) * 6)
lgd_true = np.where(cured == 1, rng.uniform(0, 0.005, n3), severity)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(lgd_true, bins=40, color="#c0562a", edgecolor="white")
ax.set_xlabel("Realized LGD")
ax.set_ylabel("Number of defaulted loans")
ax.set_title("LGD is bimodal: a cure spike near 0, a spread of partial/total losses")
plt.tight_layout()
plt.savefig(f"{OUT}/lgd_fig5_bimodal.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 6: calibration - predicted vs realized LGD (cure/severity model)
# ---------------------------------------------------------------------------
Xc = np.column_stack([ltv, liquidity])
csm = CureSeverityModel(lr=0.3, n_iter=4000, l2=1.0).fit(Xc, lgd_true)
lgd_pred = csm.predict_lgd(Xc)

bins = np.quantile(lgd_pred, np.linspace(0, 1, 11))
bin_idx = np.clip(np.digitize(lgd_pred, bins[1:-1]), 0, 9)
mean_pred = [lgd_pred[bin_idx == i].mean() for i in range(10)]
mean_real = [lgd_true[bin_idx == i].mean() for i in range(10)]

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 0.6], [0, 0.6], "--", color="gray", label="Perfect calibration")
ax.scatter(mean_pred, mean_real, color="#2a5db0", s=60, zorder=3, label="Decile bins")
ax.set_xlabel("Mean predicted LGD (decile)")
ax.set_ylabel("Mean realized LGD (decile)")
ax.set_title("Calibration: predicted vs realized LGD by decile")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/lgd_fig6_calibration.png", bbox_inches="tight")
plt.close()

print("all figures written to", OUT)
