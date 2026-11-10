"""
Post 4 -- where the weights start decides whether a network learns at all.

The frog network from part 3, trained exactly the same way six times. The only
thing that changes is the starting weights. Then a 10-layer network, where
the difference grows.

    python ../../../data/make_frogs.py    # once: builds the photo files
    python main.py
"""

import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data"
np.seterr(over="ignore")                  # exp() of a huge vote is infinity, and that's fine here


# --- load: 3,072 pixel values per photo -> frog (1) or not (0) -------------
def load(name):
    raw = np.fromfile(DATA / name, dtype=np.uint8).reshape(-1, 1 + 3072)
    return raw[:, 1:] / 255, raw[:, 0].astype(float)


X_train, y_train = load("frogs_train.bin")
X_test, y_test = load("frogs_test.bin")


def total(a):
    """Add up along the first axis strictly left to right, like the C# loops.
    numpy's own sum adds in a different order, which changes the last digits,
    and over thousands of steps the two versions would drift apart."""
    out = a[0].copy()
    for row in a[1:]:
        out += row
    return out


# centre every pixel: subtract its average over the training photos
mean = sum(X_train) / len(X_train)       # Python's sum: one photo at a time
X_train = X_train - mean
X_test = X_test - mean


class Random:
    """A tiny random generator (SplitMix64), written by hand so the C# version
    makes exactly the same choices."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self):                     # a number between -1 and 1
        return (self.next() >> 11) / 2.0**53 * 2 - 1

    def below(self, n):                    # a whole number from 0 to n - 1
        return self.next() % n


def dot(A, W, b):
    """A @ W + b, adding the products one at a time in the same order as C#."""
    out = []
    for chunk in range(0, len(A), 128):    # 128 photos at a time, to save memory
        a = A[chunk:chunk + 128]
        terms = a.T[:, :, None] * W[:, None, :]
        out.append(total(np.concatenate([np.broadcast_to(b, (1, len(a), len(b))), terms])))
    return np.concatenate(out)


# --- the six ways to start ----------------------------------------------------
# each one: a layer with n inputs -> how to pick one starting weight
STARTS = {
    "all zeros": lambda rng, n: 0.0,
    "all equal (0.01)": lambda rng, n: 0.01,             # the libreta: same ration for everyone
    "random, tiny (0.01)": lambda rng, n: rng.uniform() * 0.01,
    "random, big (1)": lambda rng, n: rng.uniform(),
    "random, 1/sqrt(n)": lambda rng, n: rng.uniform() * (1 / math.sqrt(n)),   # PyTorch's default
    "random, sqrt(6/n)": lambda rng, n: rng.uniform() * math.sqrt(6 / n),     # ours, since part 3
}


class Network:
    """sizes = [3072, 20, 7, 5, 1]: inputs, hidden layers, one output."""

    def __init__(self, sizes, start, seed=42):
        rng = Random(seed)
        self.W, self.b = [], []
        for n, h in zip(sizes, sizes[1:]):
            self.W.append(np.array([[start(rng, n) for _ in range(h)] for _ in range(n)]))
            self.b.append(np.zeros(h))

    def forward(self, X):
        """Every layer's output, from the photo (A[0]) to the answer (A[-1])."""
        A = [X]
        for l in range(len(self.W)):
            Z = dot(A[-1], self.W[l], self.b[l])
            last = l == len(self.W) - 1
            A.append(1 / (1 + np.exp(-Z)) if last else np.where(Z > 0, Z, 0.0))   # sigmoid or ReLU
        return A

    def backward(self, A, y):
        """The gradient of the cost for every weight, last layer first."""
        m = len(y)
        dZ = A[-1] - y[:, None]            # how wrong each answer was
        grads = [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            dW = total(A[l][:, :, None] * dZ[:, None, :]) / m    # blame for each weight
            db = total(dZ) / m
            grads[l] = (dW, db)
            if l > 0:
                dA = total(np.moveaxis(dZ[:, None, :] * self.W[l][None], 2, 0))
                dZ = dA * np.where(A[l] > 0, 1.0, 0.0)
        return grads

    def train(self, X, y, epochs=40, batch=64, learning_rate=0.003):
        rng = Random(7)
        order = list(range(len(y)))
        for _ in range(epochs):
            for i in reversed(range(1, len(order))):      # shuffle (Fisher-Yates)
                j = rng.below(i + 1)
                order[i], order[j] = order[j], order[i]
            for start in range(0, len(order), batch):
                rows = order[start:start + batch]
                grads = self.backward(self.forward(X[rows]), y[rows])
                for l, (dW, db) in enumerate(grads):
                    self.W[l] -= learning_rate * dW
                    self.b[l] -= learning_rate * db

    def accuracy(self, X, y):
        return 100 * np.mean((self.forward(X)[-1][:, 0] > 0.5) == (y == 1))

    def signal(self, X):
        """How big each hidden layer's outputs are (root mean square)."""
        sizes = []
        for a in self.forward(X)[1:-1]:
            squares = (a * a).ravel()      # photo by photo, neuron by neuron
            sizes.append(math.sqrt(total(squares) / len(squares)))
        return sizes


if __name__ == "__main__":
    print(f"{len(y_train)} training photos, {len(y_test)} test photos, half of each are frogs")
    for title, sizes, starts in [
        ("3 hidden layers (20, 7, 5)", [3072, 20, 7, 5, 1], list(STARTS)),
        ("10 hidden layers of 20", [3072] + [20] * 10 + [1],
         ["random, tiny (0.01)", "random, 1/sqrt(n)", "random, sqrt(6/n)"]),
    ]:
        print(f"\n{title}")
        print(f"{'starting weights':<24}{'train':>8}{'test':>8}   signal at the start, layer by layer")
        for name in starts:
            net = Network(sizes, STARTS[name])
            signal = net.signal(X_train[:500])
            net.train(X_train, y_train)
            print(f"{name:<24}{net.accuracy(X_train, y_train):>7.1f}%{net.accuracy(X_test, y_test):>7.1f}%   "
                  + " ".join(f"{s:.0e}" for s in signal))
