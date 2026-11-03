"""
The figure for post 1: every penguin, and the line where the trained neuron
switches its answer.

    python plot.py      # -> posts/images/01-neuron.png
"""

import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

m = runpy.run_path(str(Path(__file__).with_name("main.py")))
X, y, w, b, mean, std = m["X"], m["y"], m["w"], m["b"], m["mean"], m["std"]
OUT = Path(__file__).resolve().parents[3] / "posts" / "images" / "01-neuron.png"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(7, 4.6))
for label, name, color, marker in ((0, "Adelie", "#eb6834", "o"), (1, "Chinstrap", "#6250d6", "^")):
    pts = X[y == label]
    ax.scatter(pts[:, 0], pts[:, 1], s=36, c=color, marker=marker, edgecolors="white", linewidths=1, label=name, zorder=3)

# the neuron says "Chinstrap" when w0*(L-m0)/s0 + w1*(D-m1)/s1 + b > 0; solve for D
L = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 2)
D = m["mean"][1] - std[1] / w[1] * (w[0] * (L - mean[0]) / std[0] + b)
ax.plot(L, D, color=INK, lw=2, zorder=2, clip_on=True)
# label the line where it crosses a bill depth of 20.5 mm, inside the view
L_lab = mean[0] + std[0] / w[0] * (-b - w[1] * (20.5 - mean[1]) / std[1])
ax.text(L_lab + 0.4, 20.5, "the neuron's line", color=MUTED, fontsize=9, va="center", clip_on=True)
ax.text(36.4, 21.1, "Adelie", color=INK, fontsize=10, weight="bold")
ax.text(52.6, 16.3, "Chinstrap", color=INK, fontsize=10, weight="bold")

ax.set(xlabel="bill length (mm)", ylabel="bill depth (mm)", ylim=(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5),
       xlim=(L[0], L[1]))
ax.grid(color=GRID, lw=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("219 penguins and one neuron", loc="left", color=INK, fontsize=12)
OUT.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")
