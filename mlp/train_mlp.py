# mlp/train_mlp.py
import numpy as np
from .mlp_from_scratch import MLP

def simple_split(X, y, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - val_ratio))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]

def train(model, X_train, y_train, X_val, y_val,
          epochs=10, batch=64, lr=1e-3, reg=1e-4, verbose=True):
    N = X_train.shape[0]
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for e in range(epochs):
        perm = np.random.permutation(N)
        X_sh = X_train[perm]
        y_sh = y_train[perm]

        # mini-batch SGD
        for i in range(0, N, batch):
            Xb = X_sh[i:i+batch]
            yb = y_sh[i:i+batch]
            probs, cache = model.forward(Xb)
            loss, grads = model.compute_loss_and_grads(cache, yb, reg=reg)
            model.update_params(grads, lr=lr)

        # métricas al final de la época
        train_probs, train_cache = model.forward(X_train)
        train_loss, _ = model.compute_loss_and_grads(train_cache, y_train, reg=reg)
        val_probs, val_cache = model.forward(X_val)
        val_loss, _ = model.compute_loss_and_grads(val_cache, y_val, reg=reg)

        train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == y_val)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f"Epoch {e+1}/{epochs} - train_acc: {train_acc:.4f} val_acc: {val_acc:.4f} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

    return history

# simple demo runner for direct execution
if __name__ == "__main__":
    from ..data.load_mnist import load_mnist_subset
    X, y = load_mnist_subset(n_train=2000, n_test=500, normalize=True, seed=0)
    X_train, y_train, X_val, y_val = simple_split(X, y, val_ratio=0.2, seed=0)
    D = X_train.shape[1]
    C = len(np.unique(y))
    model = MLP(D, h=64, C=C, seed=1)
    train(model, X_train, y_train, X_val, y_val, epochs=10, batch=64, lr=1e-3)
