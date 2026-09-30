"""
Render the hand-worked example and the vanishing-gradient result as a
table image, matching the pattern used elsewhere in this series.
"""
import matplotlib.pyplot as plt

rows = [
    ["Item", "Value"],
    ["Input sequence x (utilization)", "[0.20, 0.40, 0.70, 0.90]"],
    ["h_1", "[0.0997, 0.0599]"],
    ["h_2", "[0.2184, 0.1644]"],
    ["h_3", "[0.3840, 0.3039]"],
    ["h_4 (final)", "[0.4999, 0.4252]"],
    ["Logits", "[0.385, 0.063]"],
    ["Softmax probs", "[0.5799, 0.4201]"],
    ["True label y", "[0, 1] (distress trend)"],
    ["Cross-entropy loss", "0.8673"],
    ["||dh_4|| (gradient at last step)", "0.5501"],
    ["||dh_1|| (gradient at first step)", "0.0054  (~100x smaller)"],
]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.axis("off")
table = ax.table(cellText=rows, cellLoc="left", loc="center", colWidths=[0.55, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for j in range(2):
    table[(0, j)].set_facecolor("#cc4125")
    table[(0, j)].set_text_props(color="white", weight="bold")
plt.title("RNN hand-worked example: every stage, verified in code", pad=20)
plt.tight_layout()
plt.savefig("images/rnn_hand_example_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Table image written to images/rnn_hand_example_table.png")
