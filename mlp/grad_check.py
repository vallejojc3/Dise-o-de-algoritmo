# mlp/grad_check.py
import numpy as np
from .mlp_from_scratch import MLP
from data.load_mnist import load_mnist_subset


def grad_check(model, X, y, reg=0.0, eps=1e-5, tol=1e-6):
    probs, cache = model.forward(X)
    _, grads = model.compute_loss_and_grads(cache, y, reg)

    params = {"W1": model.W1, "b1": model.b1, "W2": model.W2, "b2": model.b2}

    for name, param in params.items():
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original = float(param[idx])

            param[idx] = original + eps
            _, c1 = model.forward(X)
            loss1, _ = model.compute_loss_and_grads(c1, y, reg)

            param[idx] = original - eps
            _, c2 = model.forward(X)
            loss2, _ = model.compute_loss_and_grads(c2, y, reg)

            num_grad = (loss1 - loss2) / (2*eps)
            ana_grad = grads[name][idx]

            param[idx] = original
            rel_err = abs(num_grad - ana_grad) / max(1e-8, abs(num_grad) + abs(ana_grad))
            if rel_err > tol:
                print(f"Grad check FAIL on {name}{idx}: num={num_grad}, ana={ana_grad}, rel={rel_err}")
                return False

            it.iternext()

    print("Grad check PASSED")
    return True

if __name__ == "__main__":
    # mini dataset
    X, y = load_mnist_subset(n_train=200, n_test=50, normalize=True, seed=1)
    # take tiny batch for checking
    Xb = X[:5]
    yb = y[:5]
    D = X.shape[1]
    C = len(np.unique(y))
    model = MLP(D, h=10, C=C, seed=42)
    grad_check(model, Xb, yb, reg=1e-4)
