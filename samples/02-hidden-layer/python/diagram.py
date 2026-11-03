"""
The hidden-layer diagram for post 2: Moa going through the trained 3-neuron
network (the same one as 02-three-lines.png), with the real numbers.

    python diagram.py      # -> posts/images/02-network-diagram.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

import main

_, _, (W1, b1, W2, b2) = main.train(3)
raw = np.array([-74.94, 20.66])                  # Moa
x = (raw - main.mean) / main.std
z1 = x @ W1 + b1
a1 = main.tanh(z1)
z2 = a1 @ W2 + b2
p = main.sigmoid(z2)
OUT = Path(__file__).resolve().parents[3] / "posts" / "images" / "02-network-diagram.png"

INK, MUTED, LINE, FILL = "#0b0b0b", "#52514e", "#c3c2b7", "#f4f3ef"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]       # same as the three dashed lines
f = lambda v: f"{v:+.2f}".replace("-0.00", "+0.00")

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 11), ax.set_ylim(-0.25, 5.7), ax.axis("off"), ax.set_aspect("equal")


def box(cx, cy, w_, h, text, color=INK, fill=FILL, size=10):
    ax.add_patch(FancyBboxPatch((cx - w_ / 2, cy - h / 2), w_, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fill, ec=LINE, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=color, linespacing=1.5)


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


# inputs
in_y = [3.6, 1.6]
for name, r, xi, y in zip(["longitude", "latitude"], raw, x, in_y):
    box(1.15, y, 1.9, 0.95, f"{name}\n{r:.2f}°  →  {f(xi)}")
ax.text(1.15, 0.6, "Moa, Holguín", ha="center", fontsize=10, color=INK, weight="bold")

# hidden layer
hid_x, hid_y, R = 4.3, [4.6, 2.6, 0.6], 0.72
for j, y in enumerate(hid_y):
    for yi in in_y:
        arrow(2.1, yi, hid_x - R - 0.05, y + (yi - y) * 0.15, color=LINE, lw=1.0)
    ax.add_patch(Circle((hid_x, y), R, fc="white", ec=COLORS[j], lw=2.2))
    ax.text(hid_x, y + 0.3, f"hidden {j + 1}", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(hid_x, y, f"vote {f(z1[j])}", ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(hid_x, y - 0.3, f"bent {f(a1[j])}", ha="center", va="center", fontsize=9.5, color=INK, weight="bold")
ax.text(hid_x, 5.5, "hidden layer: each votes, then tanh bends it", ha="center", fontsize=9, color=MUTED)

# hidden -> output
out_x, out_y, RO = 7.6, 2.6, 0.8
for j, y in enumerate(hid_y):
    arrow(hid_x + R + 0.05, y, out_x - RO - 0.05, out_y + (y - out_y) * 0.2)
    # label next to the arrow, a little after it leaves the hidden neuron
    x0, y0, x1, y1 = hid_x + R + 0.05, y, out_x - RO - 0.05, out_y + (y - out_y) * 0.2
    t = 0.35
    below = y < out_y                              # rising arrow: put its label underneath
    ax.text(x0 + t * (x1 - x0), y0 + t * (y1 - y0) + (-0.15 if below else 0.15),
            f"× {f(W2[j])} = {f(a1[j] * W2[j])}", ha="left", va="top" if below else "bottom", fontsize=9, color=INK)
ax.add_patch(Circle((out_x, out_y), RO, fc="white", ec=INK, lw=1.8))
ax.text(out_x, out_y + 0.32, "output", ha="center", va="center", fontsize=9, color=MUTED)
ax.text(out_x, out_y, "add up", ha="center", va="center", fontsize=10, color=INK)
ax.text(out_x, out_y - 0.3, f"+ bias {f(b2)}", ha="center", va="center", fontsize=9, color=MUTED)

arrow(out_x + RO + 0.05, out_y, 9.25, out_y)
ax.text((out_x + RO + 9.25) / 2, out_y + 0.12, f"{z2:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
box(10.1, out_y, 1.45, 1.1, f"sigmoid\np = {p:.2f}\nland", size=10)

ax.set_title("Moa through the 3-neuron network", loc="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}  (z2 = {z2:.4f}, p = {p:.4f})")
