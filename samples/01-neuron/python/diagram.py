"""
The neuron diagram for post 1: the penguin from the intro (46 mm x 17 mm)
going through the trained neuron, with the real numbers.

    python diagram.py      # -> posts/images/01-neuron-diagram.png
"""

import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

m = runpy.run_path(str(Path(__file__).with_name("main.py")))
w, b, mean, std, sigmoid = m["w"], m["b"], m["mean"], m["std"], m["sigmoid"]
raw = np.array([46.0, 17.0])
x = (raw - mean) / std
z = x @ w + b
p = sigmoid(z)
OUT = Path(__file__).resolve().parents[3] / "posts" / "images" / "01-neuron-diagram.png"

INK, MUTED, LINE, FILL = "#0b0b0b", "#52514e", "#c3c2b7", "#f4f3ef"
CHINSTRAP = "#6250d6"

fig, ax = plt.subplots(figsize=(10, 4.4))
ax.set_xlim(0, 10), ax.set_ylim(0, 4.4), ax.axis("off"), ax.set_aspect("equal")


def box(cx, cy, w_, h, text, color=INK, fill=FILL, size=10):
    ax.add_patch(FancyBboxPatch((cx - w_ / 2, cy - h / 2), w_, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fill, ec=LINE, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=color, linespacing=1.5)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))


# inputs
names = ["bill length", "bill depth"]
ys = [3.2, 1.2]
for name, r, xi, wi, y in zip(names, raw, x, w, ys):
    box(1.25, y, 2.1, 1.0, f"{name}\n{r:.0f} mm  →  {xi:+.2f}", size=10)
    arrow(2.35, y, 4.55, 2.2 + (y - 2.2) * 0.25)
    label_y = y + 0.05 if y > 2.2 else y - 0.05      # just above the top arrow, below the bottom one
    ax.text(3.45, label_y, f"× weight {wi:+.2f}  =  {xi * wi:+.2f}", ha="center",
            va="bottom" if y > 2.2 else "top", fontsize=9.5, color=INK)
ax.text(1.25, 0.25, "standardized: (mm − average) / spread", ha="center", fontsize=8, color=MUTED)

# the neuron
ax.add_patch(Circle((5.4, 2.2), 0.85, fc="white", ec=INK, lw=1.8))
ax.text(5.4, 2.42, "add up", ha="center", va="center", fontsize=10, color=INK)
ax.text(5.4, 1.98, f"+ bias {b:+.2f}", ha="center", va="center", fontsize=9, color=MUTED)
ax.text(5.4, 3.35, "the neuron", ha="center", fontsize=9, color=MUTED)
arrow(6.25, 2.2, 7.0, 2.2)
ax.text(6.62, 2.38, f"z = {z:.2f}", ha="center", fontsize=9.5, color=INK)

# sigmoid and answer
box(7.65, 2.2, 1.2, 0.9, "sigmoid", size=10)
arrow(8.25, 2.2, 8.75, 2.2)
box(9.35, 2.2, 1.15, 1.2, f"p = {p:.2f}\nChinstrap", color=CHINSTRAP, fill="white", size=10.5)

ax.set_title("One penguin through the trained neuron", loc="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}  (z = {z:.4f}, p = {p:.4f})")
