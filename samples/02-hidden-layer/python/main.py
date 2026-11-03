"""
Post 2 -- one neuron can't draw Cuba. A hidden layer can.

Points around Cuba, half on land and half at sea. We train a single neuron,
a hidden layer without a bend, and hidden layers of different sizes with one.

    python main.py
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data" / "cuba.csv"

# --- load: longitude, latitude -> land (1) or sea (0) --------------------
d = np.loadtxt(DATA, delimiter=",", skiprows=1)
X, y = d[:, :2], d[:, 2]

test = np.arange(len(y)) % 4 == 3             # every 4th point is the exam
X_train, y_train, X_test, y_test = X[~test], y[~test], X[test], y[test]


def total(a):
    """Add up along the first axis strictly left to right, like the C# loops.
    numpy's own sum adds in a different order, which changes the last digits,
    and over 20,000 steps the two versions would drift apart."""
    return np.cumsum(a, axis=0)[-1]


mean = total(X_train) / len(X_train)
std = np.sqrt(total((X_train - mean) * (X_train - mean)) / len(X_train))
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


class Random:
    """A tiny random generator (SplitMix64), written by hand so the C# version
    starts from exactly the same weights."""

    def __init__(self, seed):
        self.state = seed

    def uniform(self):                         # a number between -1 and 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        return (z >> 11) / 2.0**53 * 2 - 1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    # same as np.tanh, written with exp: numpy's tanh and .NET's differ in the
    # last digit for about 1 input in 5, while their exp agrees exactly
    return 1 - 2 / (np.exp(2 * z) + 1)


def train(hidden, bend=True, steps=20_000, learning_rate=1.0):
    """hidden = 0 is a single neuron (post 1). bend=False removes tanh."""
    rng = Random(42)
    W1 = np.array([[rng.uniform() for _ in range(hidden)] for _ in range(2)])
    b1 = np.zeros(hidden)
    W2 = np.array([rng.uniform() for _ in range(hidden)]) if hidden else np.zeros(2)
    b2 = 0.0
    n = len(y_train)

    def forward(X):
        if hidden == 0:
            return X, sigmoid(X[:, 0] * W2[0] + X[:, 1] * W2[1] + b2)
        Z1 = X[:, :1] * W1[0] + X[:, 1:] * W1[1] + b1    # each hidden neuron's vote
        A1 = tanh(Z1) if bend else Z1                    # ...bent by tanh, or not
        Z2 = total(np.vstack([np.full(len(X), b2), (A1 * W2).T]))
        return A1, sigmoid(Z2)                           # the output neuron votes on their votes

    for _ in range(steps):
        A1, p = forward(X_train)
        dz2 = p - y_train                      # how wrong the output was
        if hidden:
            # blame flows back: through W2, then through the tanh's slope
            dz1 = dz2[:, None] * W2 * ((1 - A1 * A1) if bend else 1)
            W1[0] -= learning_rate * total(X_train[:, :1] * dz1) / n
            W1[1] -= learning_rate * total(X_train[:, 1:] * dz1) / n
            b1 -= learning_rate * total(dz1) / n
        W2 -= learning_rate * total(A1 * dz2[:, None]) / n
        b2 -= learning_rate * total(dz2) / n

    accuracy = lambda X, y: 100 * np.mean((forward(X)[1] > 0.5) == (y == 1))
    return accuracy(X_train, y_train), accuracy(X_test, y_test), (W1, b1, W2, b2)


if __name__ == "__main__":
    print(f"{len(y_train)} training points, {len(y_test)} test points, "
          f"{100 * y_test.mean():.1f}% of the test points are land\n")
    print(f"{'model':<28}{'train':>8}{'test':>8}")
    for label, hidden, bend in [("one neuron", 0, True), ("10 hidden, no bend", 10, False)] + \
                               [(f"{h} hidden, tanh", h, True) for h in (1, 2, 3, 4, 6, 10, 20, 50)]:
        tr, te, _ = train(hidden, bend)
        print(f"{label:<28}{tr:>7.1f}%{te:>7.1f}%")
