import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pd_scratch import woe_iv

df = pd.read_csv("data/customers8.csv")
y = df["default"].values
age_bin = (df["age"].values >= 30).astype(int)
inc_bin = (df["income_k"].values >= 32.5).astype(int)
age_woe, age_iv = woe_iv(age_bin, y, smoothing=0.5)
inc_woe, inc_iv = woe_iv(inc_bin, y, smoothing=0.0)

rows = []
rows.append(["Age < 30", age_woe[0]["events"], age_woe[0]["nonevents"],
             f"{age_woe[0]['pct_events']*100:.1f}%", f"{age_woe[0]['pct_nonevents']*100:.1f}%",
             f"{age_woe[0]['woe']:.4f}", f"{age_woe[0]['iv']:.4f}"])
rows.append(["Age >= 30", age_woe[1]["events"], age_woe[1]["nonevents"],
             f"{age_woe[1]['pct_events']*100:.1f}%", f"{age_woe[1]['pct_nonevents']*100:.1f}%",
             f"{age_woe[1]['woe']:.4f}", f"{age_woe[1]['iv']:.4f}"])
rows.append([f"Age total IV", "", "", "", "", "", f"{age_iv:.4f}"])
rows.append(["Income < 32.5k", inc_woe[0]["events"], inc_woe[0]["nonevents"],
             f"{inc_woe[0]['pct_events']*100:.1f}%", f"{inc_woe[0]['pct_nonevents']*100:.1f}%",
             f"{inc_woe[0]['woe']:.4f}", f"{inc_woe[0]['iv']:.4f}"])
rows.append(["Income >= 32.5k", inc_woe[1]["events"], inc_woe[1]["nonevents"],
             f"{inc_woe[1]['pct_events']*100:.1f}%", f"{inc_woe[1]['pct_nonevents']*100:.1f}%",
             f"{inc_woe[1]['woe']:.4f}", f"{inc_woe[1]['iv']:.4f}"])
rows.append([f"Income total IV", "", "", "", "", "", f"{inc_iv:.4f}"])

cols = ["Bin", "Events\n(defaults)", "Non-events", "%Events", "%Non-events", "WOE", "IV contrib."]

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.7)

# highlight the winner (higher total IV = Age) and the smoothing note rows
for j in range(len(cols)):
    tbl[(0, j)].set_facecolor("#2a5db0")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")
for j in range(len(cols)):
    tbl[(3, j)].set_facecolor("#eaf7ea")   # Age total IV row
for r in (1, 2, 3):
    tbl[(r, 0)].set_facecolor("#f5f5f5") if r != 3 else None

ax.set_title("WOE / IV worked by hand — Age (IV=1.903, smoothed) beats Income (IV=0.970)",
              fontsize=11, pad=14)
plt.tight_layout()
plt.savefig("images/pd_woe_iv_table.png", bbox_inches="tight", dpi=150)
plt.close()
print("saved images/pd_woe_iv_table.png")
