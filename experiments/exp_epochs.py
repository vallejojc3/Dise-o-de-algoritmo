# experiments/exp_epochs.py

import time
from data.load_mnist import load_mnist_subset
from mlp.mlp_from_scratch import MLP
from experiments.exp_batch_size import train

def experiment():
    X_train, y_train, X_val, y_val = load_mnist_subset(
        n_train=2000, n_test=500, normalize=True, seed=0
    )

    results = {}
    epoch_values = [5, 10, 20, 30]

    for E in epoch_values:
        model = MLP(784, 64, 10)  # ← CORRECTO
        t0 = time.time()
        history = train(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=E, batch=32,
            lr=1e-3, reg=1e-4,
            verbose=False
        )
        elapsed = time.time() - t0
        val_acc = history["val_acc"][-1]
        print(f"E={E} epochs -> time={elapsed:.2f}s val_acc={val_acc:.4f}")
        results[E] = (elapsed, val_acc)

    return results

if __name__ == "__main__":
    experiment()
