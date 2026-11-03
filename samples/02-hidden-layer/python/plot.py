"""
Figures for post 2.

    python plot.py      # -> posts/images/02-maps.png, 02-three-lines.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main

ROOT = Path(__file__).resolve().parents[3]
OUTLINE = [np.array(r) for r in json.load(open(ROOT / "data" / "cuba_outline.json"))]
INK, MUTED, LAND, SEA = "#0b0b0b", "#52514e", "#d8cfb8", "#f3f6f8"
LINES = ["#2a78d6", "#eb6834", "#1baf7a"]
DASHES = ["--", "-.", ":"]

# a grid of map points, standardized the same way as the training data
lon, lat = np.meshgrid(np.linspace(-85.5, -73.5, 400), np.linspace(19.3, 23.7, 160))
grid = (np.column_stack([lon.ravel(), lat.ravel()]) - main.mean) / main.std


def predict(params, hidden, bend):
    W1, b1, W2, b2 = params
    if hidden == 0:
        z = grid @ W2 + b2
    else:
        a = grid @ W1 + b1
        z = (np.tanh(a) if bend else a) @ W2 + b2
    return (z > 0).reshape(lon.shape)


def draw(ax, land, title):
    ax.contourf(lon, lat, land, levels=[-0.5, 0.5, 1.5], colors=[SEA, LAND], alpha=0.9)
    for ring in OUTLINE:
        ax.plot(ring[:, 0], ring[:, 1], color=INK, lw=0.8)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect(1 / np.cos(np.radians(21.5)))
    for s in ax.spines.values():
        s.set_color("#c3c2b7")


configs = [("one neuron", 0, True), ("10 hidden, no bend", 10, False), ("1 hidden", 1, True),
           ("2 hidden", 2, True), ("3 hidden", 3, True), ("6 hidden", 6, True), ("20 hidden", 20, True),
           ("50 hidden", 50, True)]
fig, axes = plt.subplots(4, 2, figsize=(10, 7.6))
for ax, (label, h, bend) in zip(axes.flat, configs):
    _, test, params = main.train(h, bend)
    draw(ax, predict(params, h, bend), f"{label}: {test:.1f}% of test points right")
    if h == 3:
        three = params
fig.suptitle("Where each model thinks the land is (shaded), against the real coastline",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(ROOT / "posts" / "images" / "02-maps.png", dpi=110)

# the three lines a 3-unit hidden layer draws, and what the output makes of them
W1, b1, W2, b2 = three
fig, ax = plt.subplots(figsize=(10, 4.2))
draw(ax, predict(three, 3, True), "3 hidden neurons: each draws one line; the output neuron combines them")
xs = np.linspace(-85.5, -73.5, 200)
for j in range(3):
    # the line where hidden neuron j's vote is zero, back in longitude/latitude
    w_lon, w_lat = W1[0, j] / main.std[0], W1[1, j] / main.std[1]
    c = b1[j] - w_lon * main.mean[0] - w_lat * main.mean[1]
    ys = -(w_lon * xs + c) / w_lat
    ax.plot(xs, ys, color=LINES[j], lw=2.2, ls=DASHES[j], label=f"hidden neuron {j + 1}")
ax.set_ylim(19.3, 23.7), ax.set_xlim(-85.5, -73.5)
ax.legend(loc="lower left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "posts" / "images" / "02-three-lines.png", dpi=110)
print("saved 02-maps.png, 02-three-lines.png")
