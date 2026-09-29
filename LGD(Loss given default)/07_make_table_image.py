import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lgd_scratch import workout_lgd_r

df = pd.read_csv("data/defaulted_loans3.csv")
rows = []
for _, row in df.iterrows():
    res = workout_lgd_r(row["EAD"], [(row["t_years"], row["recovery"], row["cost"])], r=0.10)
    d = res["detail"][0]
    rows.append([
        f"Cust {row['customer']} ({row['loan_type'].replace('_',' ')})",
        f"{row['EAD']:,.0f}",
        f"{row['t_years']:.2f}y",
        f"{row['recovery']:,.0f}",
        f"{row['cost']:,.0f}",
        f"{d['net']:,.0f}",
        f"{d['pv']:,.0f}",
        f"{res['lgd']*100:.2f}%",
    ])

cols = ["Loan", "EAD", "Time to\nresolution", "Recovery", "Cost", "Net\ncashflow", "PV @ r=10%", "LGD"]

fig, ax = plt.subplots(figsize=(11, 2.6))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.9)

for j in range(len(cols)):
    tbl[(0, j)].set_facecolor("#c0562a")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

ax.set_title("Workout LGD worked by hand for all three defaulted loans (r=10%)",
              fontsize=11, pad=14)
plt.tight_layout()
plt.savefig("images/lgd_workout_table.png", bbox_inches="tight", dpi=150)
plt.close()
print("saved images/lgd_workout_table.png")
