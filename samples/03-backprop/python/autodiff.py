"""
Post 3, last part -- let PyTorch do the backward pass.

1. Same network, same starting weights, same 16 photos: PyTorch's gradients
   against the ones from our hand-written backward().
2. The same training run, written the way people write it with PyTorch.

    pip install torch
    python autodiff.py
"""

import time

import numpy as np
import torch

from main import Network, X_test, X_train, y_test, y_train

torch.set_default_dtype(torch.float64)    # same precision as our numpy version
sizes = [3072, 20, 7, 5, 1]

# --- 1. our gradients vs. PyTorch's ----------------------------------------
ours = Network(sizes)
X, y = X_train[:16], y_train[:16]
our_grads = ours.backward(ours.forward(X), y)

# the same weights, as tensors PyTorch should track
W = [torch.tensor(w, requires_grad=True) for w in ours.W]
b = [torch.tensor(v, requires_grad=True) for v in ours.b]
a = torch.tensor(X)
for l in range(len(W)):
    z = a @ W[l] + b[l]
    a = torch.sigmoid(z) if l == len(W) - 1 else torch.relu(z)
p = a[:, 0]
t = torch.tensor(y)
cost = -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()
cost.backward()                           # <- the whole backward pass

worst = max(np.max(np.abs(g.grad.numpy() - mine)) / np.max(np.abs(mine))
            for g, mine in zip(W + b, [dW for dW, _ in our_grads] + [db for _, db in our_grads]))
print(f"largest difference, our gradients vs. PyTorch's: {worst:.0e} (relative)")

# --- 2. the usual PyTorch way -------------------------------------------------
def make_model(start_from_ours):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(3072, 20), torch.nn.ReLU(),
        torch.nn.Linear(20, 7), torch.nn.ReLU(),
        torch.nn.Linear(7, 5), torch.nn.ReLU(),
        torch.nn.Linear(5, 1),            # no sigmoid: the loss below includes it
    )
    if start_from_ours:                   # copy our starting weights in
        with torch.no_grad():
            for layer, w, v in zip(model[::2], ours_at_start.W, ours_at_start.b):
                layer.weight.copy_(torch.tensor(w.T))
                layer.bias.copy_(torch.tensor(v))
    return model


ours_at_start = Network(sizes)
Xt, yt = torch.tensor(X_train), torch.tensor(y_train)
for title, start_from_ours in [("our starting weights", True), ("PyTorch's starting weights", False)]:
    model = make_model(start_from_ours)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

    start = time.time()
    for epoch in range(40):
        for rows in torch.randperm(len(yt)).split(64):
            loss = loss_fn(model(Xt[rows])[:, 0], yt[rows])
            optimizer.zero_grad()
            loss.backward()               # backprop
            optimizer.step()              # w -= learning_rate * gradient
    seconds = time.time() - start

    with torch.no_grad():
        accuracy = lambda X, y: 100 * ((model(torch.tensor(X))[:, 0] > 0) == torch.tensor(y == 1)).double().mean()
        print(f"PyTorch, {title}: train {accuracy(X_train, y_train):.1f}%, "
              f"test {accuracy(X_test, y_test):.1f}%, trained in {seconds:.0f} s")
