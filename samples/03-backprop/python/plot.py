"""
Figures for post 3.

    python plot.py   # -> posts/images/03-photos.png, 03-blame.png, 03-nudge.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

import main

OUT = Path(__file__).resolve().parents[3] / "posts" / "images"
INK, MUTED, LINE, FILL = "#0b0b0b", "#52514e", "#c3c2b7", "#f4f3ef"
FORWARD, BLAME = "#2a78d6", "#eb6834"


def photo(raw_row):
    """3,072 bytes (all red, all green, all blue) -> a 32 x 32 x 3 picture."""
    return raw_row.reshape(3, 32, 32).transpose(1, 2, 0)


raw = np.fromfile(main.DATA / "frogs_test.bin", dtype=np.uint8).reshape(-1, 1 + 3072)

# --- 1. what the photos look like -------------------------------------------
frogs, others = np.flatnonzero(raw[:, 0] == 1)[:8], np.flatnonzero(raw[:, 0] == 0)[:8]
fig, axes = plt.subplots(2, 8, figsize=(10, 2.9))
for row, picks, label in [(0, frogs, "frog (1)"), (1, others, "not a frog (0)")]:
    for ax, i in zip(axes[row], picks):
        ax.imshow(photo(raw[i, 1:]), interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[row, 0].set_ylabel(label, fontsize=10, color=INK)
fig.suptitle("Test photos, 32 × 32 pixels each, shown big", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(OUT / "03-photos.png", dpi=130)

# --- 2. one frog, forward and backward through the trained network -----------
net = main.Network([3072, 20, 7, 5, 1])
net.train(main.X_train, main.y_train)
p_all = net.forward(main.X_test)[-1][:, 0]
frogs_all = np.flatnonzero(raw[:, 0] == 1)
fires = lambda i: all((a[0] > 0).any() for a in net.forward(main.X_test[i:i + 1])[1:-1])
pick = max((i for i in frogs_all if fires(i)), key=lambda i: p_all[i])   # its surest frog with every layer firing
A = net.forward(main.X_test[pick:pick + 1])
p = A[-1][0, 0]

# the same blame the backward pass computes, kept for every layer
blame = [A[-1] - 1.0]                                          # p - y, and y = 1 (frog)
for l in reversed(range(1, len(net.W))):
    dA = blame[0] @ net.W[l].T
    blame.insert(0, dA * (A[l] > 0))

fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 11), ax.set_ylim(0, 4.6), ax.axis("off")


def box(cx, cy, w, h, text, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                                fc=FILL, ec=LINE, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.5, weight=weight)


def arrow(x0, x1, y, color, text):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.6))
    ax.text((x0 + x1) / 2, y + (0.12 if color == FORWARD else -0.12), text, ha="center",
            va="bottom" if color == FORWARD else "top", fontsize=8.5, color=INK)


# the photo itself, as a small image
img = fig.add_axes([0.02, 0.32, 0.13, 0.32])
img.imshow(photo(raw[pick, 1:]), interpolation="nearest"), img.axis("off")
img.set_title("a test frog", fontsize=9, color=INK)

xs, y_box = [3.2, 5.4, 7.3, 9.6], 2.3
names = ["layer 1\n20 neurons", "layer 2\n7 neurons", "layer 3\n5 neurons"]
for l, (x, name) in enumerate(zip(xs[:3], names)):
    fired = int((A[l + 1][0] > 0).sum())
    size = np.sqrt((blame[l][0] ** 2).sum())
    box(x, y_box, 1.55, 1.5, f"{name}\n{fired} fired\nblame {size:.3f}")
box(xs[3], y_box, 1.55, 1.5, f"output\np = {p:.2f}\nfrog", size=10, weight="bold")

ends = [(1.75, xs[0] - 0.8)] + [(xs[i] + 0.8, xs[i + 1] - 0.8) for i in range(3)]
for i, (x0, x1) in enumerate(ends):
    arrow(x0, x1, y_box + 0.45, FORWARD, "3,072 pixels" if i == 0 else "")
for i, (x0, x1) in enumerate(ends):
    arrow(x1, x0, y_box - 0.45, BLAME, f"p − 1 = {p - 1:.2f}" if i == 3 else "")

ax.text(6.4, 3.75, "forward: the photo becomes an answer", ha="center", fontsize=10, color=FORWARD, weight="bold")
ax.text(6.4, 0.85, "backward: the miss becomes blame, layer by layer", ha="center", fontsize=10, color=BLAME,
        weight="bold")
ax.text(6.4, 0.45, "a neuron that didn't fire gets no blame, and passes none down", ha="center", fontsize=9,
        color=MUTED)
fig.savefig(OUT / "03-blame.png", dpi=130)
print(f"photo {pick}: p = {p:.4f}, fired {[int((a[0] > 0).sum()) for a in A[1:-1]]}")

# --- 3. how big a nudge ------------------------------------------------------
epsilons = [10.0 ** -e for e in range(1, 14)]
X, y = main.X_train[:16], main.y_train[:16]
diffs = [main.gradient_check([3072, 20, 7, 5, 1], X, y, e)[1] for e in epsilons]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.loglog(epsilons, diffs, color=FORWARD, lw=2, marker="o", ms=6)
ax.invert_xaxis()
best = int(np.argmin(diffs))
ax.annotate(f"best: {epsilons[best]:.0e}", (epsilons[best], diffs[best]), xytext=(0, 12),
            textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
ax.text(epsilons[0], diffs[2] / 8, "too big: the cost curves\ninside the nudge", fontsize=9, color=MUTED, va="top")
ax.text(epsilons[-1], diffs[-1] * 2.2, "too small: rounding\nnoise takes over", fontsize=9, color=MUTED,
        ha="right", va="bottom")
ax.set_xlabel("size of the nudge (ε)", color=INK)
ax.set_ylabel("difference from backprop", color=INK)
ax.set_title("Gradient check of a correct backward pass: smaller isn't always better", loc="left",
             fontsize=11, color=INK)
ax.grid(True, which="major", color="#e6e5df", lw=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "03-nudge.png", dpi=130)
print("diffs", [f"{d:.1e}" for d in diffs])
