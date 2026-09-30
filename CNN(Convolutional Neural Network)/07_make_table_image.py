"""
Render the hand-worked forward/backward pass as a table image, matching
the pd_woe_iv_table.png / lgd_workout_table.png / ead_ccf_table.png
pattern used elsewhere in this series.
"""
import matplotlib.pyplot as plt

rows = [
    ["Stage", "Shape", "Values"],
    ["Input X", "6x6", "vertical-edge block image"],
    ["Kernel K, b_conv", "3x3, scalar", "[[1,0,-1]x3], b=-1.0"],
    ["Conv output Z", "4x4", "rows of [-1, 2, 2, -1]"],
    ["After ReLU, A", "4x4", "rows of [0, 2, 2, 0]"],
    ["After max pool, P", "2x2", "all 2.0"],
    ["Flatten", "(4,)", "[2, 2, 2, 2]"],
    ["Logits", "(2,)", "[1.3, 3.1]"],
    ["Softmax probs", "(2,)", "[0.1419, 0.8581]"],
    ["True label y", "(2,)", "[0, 1]"],
    ["Cross-entropy loss", "scalar", "0.1530"],
    ["dL/dlogits = p - y", "(2,)", "[0.1419, -0.1419]"],
    ["dL/db_conv", "scalar", "-0.1419"],
]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axis("off")
table = ax.table(cellText=rows, cellLoc="left", loc="center",
                  colWidths=[0.28, 0.16, 0.56])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for j in range(3):
    table[(0, j)].set_facecolor("#4a86e8")
    table[(0, j)].set_text_props(color="white", weight="bold")
plt.title("CNN hand-worked example: every stage, verified in code", pad=20)
plt.tight_layout()
plt.savefig("images/cnn_hand_example_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Table image written to images/cnn_hand_example_table.png")
