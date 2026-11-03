"""
Post 1 -- a single neuron (logistic regression) tells Adelie from Chinstrap
penguins using two numbers: bill length and bill depth.

    python main.py
"""

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data" / "penguins.csv"

# --- load: Adelie (0) vs Chinstrap (1), rows with missing bills dropped ----
rows = [r for r in csv.DictReader(open(DATA))
        if r["species"] in ("Adelie", "Chinstrap") and r["bill_length_mm"] != "NA"]
X = np.array([[float(r["bill_length_mm"]), float(r["bill_depth_mm"])] for r in rows])
y = np.array([1.0 if r["species"] == "Chinstrap" else 0.0 for r in rows])

# every 4th penguin is held out for testing; no randomness, so C# matches exactly
test = np.arange(len(y)) % 4 == 3
X_train, y_train, X_test, y_test = X[~test], y[~test], X[test], y[test]

# --- standardize: centre each feature on 0 with spread 1 (train stats only) ----
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- train: gradient descent on the log loss -----------------------------
w, b = np.zeros(2), 0.0
learning_rate = 0.1
n = len(y_train)

for step in range(1001):
    p = sigmoid(X_train @ w + b)                 # predicted chance of Chinstrap
    loss = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
    dz = p - y_train                             # how wrong each prediction was
    w -= learning_rate * (X_train.T @ dz) / n
    b -= learning_rate * dz.mean()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")


def accuracy(X, y):
    return 100 * np.mean((sigmoid(X @ w + b) > 0.5) == (y == 1))


print(f"\nweights: bill length {w[0]:+.3f}, bill depth {w[1]:+.3f}, bias {b:+.3f}")
print(f"train accuracy {accuracy(X_train, y_train):.1f}%  ({len(y_train)} penguins)")
print(f"test  accuracy {accuracy(X_test, y_test):.1f}%  ({len(y_test)} penguins)")
print(f"always guessing Adelie: {100 * np.mean(y_test == 0):.1f}%")
