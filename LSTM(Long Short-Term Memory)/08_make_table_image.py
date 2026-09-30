"""
Render the hand-worked example and the gradient-retention comparison as
a table image, matching the pattern used elsewhere in this series.
"""
import matplotlib.pyplot as plt

rows = [
    ["Item", "Value"],
    ["Input sequence x (utilization)", "[0.20, 0.40, 0.70, 0.90]"],
    ["c_4 (final cell state)", "[0.4139, 0.3526]"],
    ["h_4 (final hidden state)", "[0.2201, 0.1823]"],
    ["Logits", "[0.1685, 0.0251]"],
    ["Softmax probs", "[0.5358, 0.4642]"],
    ["True label y", "[0, 1] (distress trend)"],
    ["Cross-entropy loss", "0.7674"],
    ["||dh_4|| (gradient at last step)", "0.5083"],
    ["||dh_1|| (gradient at first step)", "0.0125"],
    ["RNN ||dh_1|| at T=30 (same weights scale)", "0.0000045"],
    ["LSTM ||dh_1|| at T=30 (same weights scale)", "0.0057  (~1270x more)"],
]

fig, ax = plt.subplots(figsize=(9, 5.8))
ax.axis("off")
table = ax.table(cellText=rows, cellLoc="left", loc="center", colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for j in range(2):
    table[(0, j)].set_facecolor("#38761d")
    table[(0, j)].set_text_props(color="white", weight="bold")
plt.title("LSTM hand-worked example + gradient retention vs RNN, verified in code", pad=20)
plt.tight_layout()
plt.savefig("images/lstm_hand_example_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Table image written to images/lstm_hand_example_table.png")
