import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ead_scratch import ccf_from_cohort

df = pd.read_csv("data/revolving_accounts3.csv")
rows = []
for _, row in df.iterrows():
    ccf = ccf_from_cohort(row["drawn_ref"], row["limit_ref"], row["drawn_default"])
    undrawn = row["limit_ref"] - row["drawn_ref"]
    rows.append([
        f"Cust {int(row['customer'])}",
        f"{row['limit_ref']:,.0f}",
        f"{row['drawn_ref']:,.0f}",
        f"{undrawn:,.0f}",
        f"{row['drawn_default']:,.0f}",
        f"{ccf*100:.2f}%",
    ])

cols = ["Account", "Limit", "Drawn @ ref\n(12mo pre-default)", "Undrawn @ ref", "Drawn @ default", "CCF"]

fig, ax = plt.subplots(figsize=(10, 2.2))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.9)

for j in range(len(cols)):
    tbl[(0, j)].set_facecolor("#2a8f5c")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

ax.set_title("CCF worked by hand for all three historical defaulted accounts",
              fontsize=11, pad=14)
plt.tight_layout()
plt.savefig("images/ead_ccf_table.png", bbox_inches="tight", dpi=150)
plt.close()
print("saved images/ead_ccf_table.png")
