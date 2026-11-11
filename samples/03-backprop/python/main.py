"""
Post 3 -- backpropagation for any number of layers, and how to check it.

Tiny photos, frog or not. We write the backward pass once, for any depth,
train networks of different depths, then check the gradients by nudging each
weight and watching the cost move. A bug that crashes nothing gets caught.

    python ../../../data/make_frogs.py    # once: builds the photo files
    python main.py
"""

import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data"


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


# math.exp and math.log are the system's own, the same ones .NET calls. On Linux
# numpy brings its own versions, which differ in the last digit now and then:
# harmless for training, but the gradient check below is sensitive enough to see it.
exp = np.vectorize(math.exp)
log = np.vectorize(math.log)


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


class Network:
    """sizes = [3072, 20, 7, 5, 1]: inputs, hidden layers, one output."""

    def __init__(self, sizes, seed=42):
        rng = Random(seed)
        self.W, self.b = [], []
        for n, h in zip(sizes, sizes[1:]):
            scale = math.sqrt(6 / n)       # keeps signals from growing layer by layer (part 4)
            self.W.append(np.array([[rng.uniform() * scale for _ in range(h)] for _ in range(n)]))
            self.b.append(np.zeros(h))

    def forward(self, X):
        """Every layer's output, from the photo (A[0]) to the answer (A[-1])."""
        A = [X]
        for l in range(len(self.W)):
            Z = dot(A[-1], self.W[l], self.b[l])
            last = l == len(self.W) - 1
            A.append(1 / (1 + exp(-Z)) if last else np.where(Z > 0, Z, 0.0))   # sigmoid or ReLU
        return A

    def cost(self, X, y):
        p = self.forward(X)[-1][:, 0]
        return -total(y * log(p) + (1 - y) * log(1 - p)) / len(y)

    def backward(self, A, y, bug=False):
        """The gradient of the cost for every weight, last layer first."""
        m = len(y)
        dZ = A[-1] - y[:, None]            # how wrong each answer was
        grads = [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            dW = total(A[l][:, :, None] * dZ[:, None, :]) / m    # blame for each weight
            db = total(dZ) / m
            grads[l] = (dW, db)
            if l > 0:
                # pass the blame one layer down: through the weights...
                dA = total(np.moveaxis(dZ[:, None, :] * self.W[l][None], 2, 0))
                # ...and through ReLU's slope: 1 where the neuron fired, 0 where it didn't
                dZ = dA if bug else dA * np.where(A[l] > 0, 1.0, 0.0)
        return grads

    def train(self, X, y, epochs=40, batch=64, learning_rate=0.003, bug=False):
        rng = Random(7)
        order = list(range(len(y)))
        for _ in range(epochs):
            for i in reversed(range(1, len(order))):      # shuffle (Fisher-Yates)
                j = rng.below(i + 1)
                order[i], order[j] = order[j], order[i]
            for start in range(0, len(order), batch):
                rows = order[start:start + batch]
                grads = self.backward(self.forward(X[rows]), y[rows], bug)
                for l, (dW, db) in enumerate(grads):
                    self.W[l] -= learning_rate * dW
                    self.b[l] -= learning_rate * db

    def accuracy(self, X, y):
        return 100 * np.mean((self.forward(X)[-1][:, 0] > 0.5) == (y == 1))


def gradient_check(sizes, X, y, epsilon=1e-7, bug=False, weights_in_first_layer=100):
    """Backprop's gradients vs. ones measured by nudging each weight.
    Returns the relative difference for each layer and for all of them."""
    net = Network(sizes)
    grads = net.backward(net.forward(X), y, bug)
    rng = Random(3)
    claimed, measured, layer_of = [], [], []
    for l, (W, b) in enumerate(zip(net.W, net.b)):
        n, h = W.shape
        cells = ([(rng.below(n), rng.below(h)) for _ in range(weights_in_first_layer)] if l == 0
                 else [(k, j) for k in range(n) for j in range(h)])
        for target, grad, index in [(W, grads[l][0], c) for c in cells] + \
                                   [(b, grads[l][1], j) for j in range(h)]:
            keep = target[index]
            target[index] = keep + epsilon
            plus = net.cost(X, y)
            target[index] = keep - epsilon
            minus = net.cost(X, y)
            target[index] = keep
            claimed.append(grad[index])
            measured.append((plus - minus) / (2 * epsilon))
            layer_of.append(l)

    def difference(keep):
        g = np.array([c for c, k in zip(claimed, keep) if k])
        a = np.array([m for m, k in zip(measured, keep) if k])
        norm = lambda v: math.sqrt(total(v * v))
        return norm(g - a) / (norm(g) + norm(a))

    per_layer = [difference([l == i for l in layer_of]) for i in range(len(sizes) - 1)]
    return per_layer, difference([True] * len(layer_of)), len(claimed)


if __name__ == "__main__":
    print(f"{len(y_train)} training photos, {len(y_test)} test photos, half of each are frogs\n")
    print(f"{'model':<34}{'train':>8}{'test':>8}")
    for label, sizes, bug in [("one neuron", [3072, 1], False),
                              ("1 hidden layer (20)", [3072, 20, 1], False),
                              ("3 hidden layers (20, 7, 5)", [3072, 20, 7, 5, 1], False),
                              ("3 hidden layers, backprop bug", [3072, 20, 7, 5, 1], True)]:
        net = Network(sizes)
        net.train(X_train, y_train, bug=bug)
        print(f"{label:<34}{net.accuracy(X_train, y_train):>7.1f}%{net.accuracy(X_test, y_test):>7.1f}%")

    sizes, X, y = [3072, 20, 7, 5, 1], X_train[:16], y_train[:16]
    for title, bug in [("correct backprop", False), ("backprop bug", True)]:
        per_layer, overall, count = gradient_check(sizes, X, y, bug=bug)
        print(f"\ngradient check, {title} ({count} weights, 16 photos): difference {overall:.1e}")
        for l, d in enumerate(per_layer):
            print(f"  layer {l + 1}: {d:.1e}{'   <- wrong' if d > 1e-5 else ''}")

    print("\nsize of the nudge vs. difference (correct backprop)")
    for epsilon in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]:
        print(f"  {epsilon:.0e}: {gradient_check(sizes, X, y, epsilon)[1]:.1e}")
