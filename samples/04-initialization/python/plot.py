"""
Figures for post 4.

    python plot.py   # -> posts/images/04-signal.png, 04-neurons.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main

OUT = Path(__file__).resolve().parents[3] / "posts" / "images"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5df"
LINES = [("random, sqrt(6/n)", "ours: sqrt(6/n)", "#2a78d6", "-"),
         ("random, 1/sqrt(n)", "PyTorch's default: 1/sqrt(n)", "#eb6834", "--"),
         ("random, tiny (0.01)", "tiny: 0.01", "#1baf7a", ":")]

# --- 1. the signal through 10 layers, before any training ---------------------
deep = [3072] + [20] * 10 + [1]
fig, ax = plt.subplots(figsize=(8, 4.4))
for name, label, color, dash in LINES:
    signal = main.Network(deep, main.STARTS[name]).signal(main.X_train[:500])
    layers = range(1, len(signal) + 1)
    ax.semilogy(layers, signal, color=color, ls=dash, lw=2, marker="o", ms=6)
    ax.text(10.25, signal[-1], label, color=INK, fontsize=9, va="center")
ax.set_xlim(0.5, 13.2)
ax.set_xticks(range(1, 11))
ax.set_xlabel("hidden layer", color=INK)
ax.set_ylabel("size of the signal (log scale)", color=INK)
ax.set_title("The same photos through 10 untrained layers: the signal survives, fades, or dies",
             loc="left", fontsize=11, color=INK)
ax.grid(True, which="major", color=GRID, lw=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "04-signal.png", dpi=130)

# --- 2. what each first-layer neuron looks for, after training ----------------
fig = plt.figure(figsize=(10, 5.4))
for block, (name, title) in enumerate([("all equal (0.01)", "Started equal (the libreta): 20 identical neurons"),
                                       ("random, sqrt(6/n)", "Started random, sqrt(6/n): what each of the 20 neurons learned")]):
    net = main.Network([3072, 20, 7, 5, 1], main.STARTS[name])
    start = net.W[0].copy()
    net.train(main.X_train, main.y_train)
    # random starts are noise; show the change training made, which is the part learned
    W = net.W[0] if block == 0 else net.W[0] - start
    top = 0.93 - block * 0.48
    fig.text(0.01, top, title, fontsize=11, color=INK, va="bottom")
    for j in range(20):
        r, c = divmod(j, 10)
        ax = fig.add_axes([0.01 + c * 0.099, top - 0.215 * (r + 1), 0.09, 0.2])
        w = W[:, j].reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow((w - W.min()) / (W.max() - W.min()), interpolation="nearest")   # one colour scale per block
        ax.axis("off")
    same = all(np.array_equal(net.W[0][:, 0], net.W[0][:, j]) for j in range(20))
    print(f"{name}: all 20 neurons identical after training: {same}")
fig.savefig(OUT / "04-neurons.png", dpi=130)
