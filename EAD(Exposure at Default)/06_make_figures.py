import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ead_scratch import ccf_from_cohort, ead_forecast, downturn_ccf, BetaRegression
from scipy.stats import spearmanr

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
OUT = "images"

# ---------------------------------------------------------------------------
# Fig 1: EAD pipeline schematic
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.axis("off")
steps = ["Reference\ndate", "Undrawn\nlimit", "CCF\nestimate", "Forecast\nEAD", "Downturn\nadjust", "Validate"]
xs = np.linspace(0.05, 0.95, len(steps))
for i, (x, s) in enumerate(zip(xs, steps)):
    box = dict(boxstyle="round,pad=0.5", fc="#eaf5ee", ec="#2a8f5c", lw=1.5)
    ax.text(x, 0.5, s, ha="center", va="center", fontsize=10.5, bbox=box, transform=ax.transAxes)
    if i < len(steps) - 1:
        ax.annotate("", xy=(xs[i+1]-0.06, 0.5), xytext=(x+0.06, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#2a8f5c"))
ax.set_title("EAD pipeline: current draw + CCF-projected drawdown of the undrawn line", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/ead_fig1_pipeline.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 2: CCF per account + average
# ---------------------------------------------------------------------------
df = pd.read_csv("data/revolving_accounts3.csv")
ccfs = [ccf_from_cohort(r["drawn_ref"], r["limit_ref"], r["drawn_default"]) for _, r in df.iterrows()]
labels = [f"Cust {int(r['customer'])}" for _, r in df.iterrows()]
avg_ccf = np.mean(ccfs)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(labels, np.array(ccfs) * 100, color="#2a8f5c")
ax.axhline(avg_ccf * 100, color="#c0562a", ls="--", lw=1.5, label=f"Average CCF = {avg_ccf*100:.1f}%")
for b, v in zip(bars, ccfs):
    ax.text(b.get_x() + b.get_width()/2, v*100 + 1.5, f"{v*100:.1f}%", ha="center")
ax.set_ylabel("Realized CCF (%)")
ax.set_title("CCF per historical defaulted account (12-month reference window)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/ead_fig2_ccf_per_account.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 3: EAD forecast waterfall for the new applicant
# ---------------------------------------------------------------------------
limit_now, drawn_now = 80000, 30000
undrawn_now = limit_now - drawn_now
ead_lra = ead_forecast(drawn_now, limit_now, avg_ccf)
years_ccf = np.array([0.55, 0.50, 0.58, 0.82, 0.88, 0.52, 0.56, 0.60])
downturn_mask = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
dt = downturn_ccf(years_ccf, downturn_mask)
ead_downturn = ead_forecast(drawn_now, limit_now, dt["regulatory_ccf"])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
labels2 = ["Drawn\ntoday", "+ CCF x undrawn\n(LRA)", "= EAD\n(LRA)", "+ CCF x undrawn\n(downturn)", "= EAD\n(downturn)"]
vals = [drawn_now, ead_lra - drawn_now, ead_lra, ead_downturn - drawn_now, ead_downturn]
colors = ["#8a8a8a", "#2a8f5c", "#2a5db0", "#d9534f", "#c0562a"]
ax.bar(labels2, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v + 1200, f"{v:,.0f}", ha="center", fontsize=9)
ax.set_ylabel("₹")
ax.set_title("EAD forecast: drawn today + CCF x undrawn limit")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f"{OUT}/ead_fig3_forecast_waterfall.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 4: downturn vs LRA CCF across 8 years
# ---------------------------------------------------------------------------
years = [f"Y{i+1}" for i in range(8)]
colors4 = ["#d9534f" if m else "#2a5db0" for m in downturn_mask]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(years, years_ccf * 100, color=colors4)
ax.axhline(dt["lra_ccf"] * 100, color="#2a5db0", ls="--", lw=1.5, label=f"LRA CCF = {dt['lra_ccf']*100:.1f}%")
ax.axhline(dt["downturn_ccf"] * 100, color="#d9534f", ls="--", lw=1.5, label=f"Downturn CCF = {dt['downturn_ccf']*100:.1f}%")
ax.set_ylabel("Portfolio CCF (%)")
ax.set_title("Downturn years (red): distressed borrowers draw down lines faster")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/ead_fig4_downturn.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 5: SA-CCR EAD breakdown (RC vs alpha*PFE)
# ---------------------------------------------------------------------------
RC, PFE, alpha = 150000, 50000, 1.4
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.bar(["RC", "PFE", "alpha x (RC+PFE)\n= EAD"], [RC, PFE, alpha * (RC + PFE)],
       color=["#8a8a8a", "#2a8f5c", "#2a5db0"])
for i, v in enumerate([RC, PFE, alpha * (RC + PFE)]):
    ax.text(i, v + 3000, f"{v:,.0f}", ha="center")
ax.set_ylabel("₹")
ax.set_title("SA-CCR: EAD = alpha x (RC + PFE) for a derivative exposure")
plt.tight_layout()
plt.savefig(f"{OUT}/ead_fig5_saccr.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Fig 6: CCF model calibration by decile (synthetic portfolio)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
n3 = 2000
utilization = rng.uniform(0.1, 0.95, n3)
months_to_default = rng.uniform(1, 24, n3)
line_age_years = rng.uniform(0.5, 10, n3)
X3 = np.column_stack([utilization, months_to_default, line_age_years])
mu_ccf = 1 / (1 + np.exp(-(0.5 + 1.4 * utilization - 0.03 * months_to_default - 0.05 * line_age_years)))
ccf_true = rng.beta(mu_ccf * 8, (1 - mu_ccf) * 8)
ccf_model = BetaRegression().fit(X3, ccf_true)
ccf_pred = ccf_model.predict(X3)

bins = np.quantile(ccf_pred, np.linspace(0, 1, 11))
bin_idx = np.clip(np.digitize(ccf_pred, bins[1:-1]), 0, 9)
mean_pred = [ccf_pred[bin_idx == i].mean() for i in range(10)]
mean_real = [ccf_true[bin_idx == i].mean() for i in range(10)]

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
ax.scatter(mean_pred, mean_real, color="#2a8f5c", s=60, zorder=3, label="Decile bins")
ax.set_xlabel("Mean predicted CCF (decile)")
ax.set_ylabel("Mean realized CCF (decile)")
ax.set_title("CCF calibration: predicted vs realized by decile")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/ead_fig6_calibration.png", bbox_inches="tight")
plt.close()

print("all figures written to", OUT)
