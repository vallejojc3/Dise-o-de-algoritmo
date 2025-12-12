# experiments/exp_hidden_size.py
import time
import numpy as np
from mlp.mlp_from_scratch import MLP
from mlp.train_mlp import train, simple_split
from data.load_mnist import load_mnist_subset

def experiment(h_list=[16,32,64,128], epochs=5):
    X_train_all, y_train_all, X_test, y_test = load_mnist_subset(n_train=2000, n_test=500, normalize=True, seed=0)
    results = {}
    X_train, y_train, X_val, y_val = simple_split(X_train_all, y_train_all, val_ratio=0.2, seed=0)
    D = X_train.shape[1]
    C = len(np.unique(y_train))
    for h in h_list:
        model = MLP(D, h=h, C=C, seed=1)
        t0 = time.time()
        history = train(model, X_train, y_train, X_val, y_val, epochs=epochs, batch=64, lr=1e-3, verbose=False)
        t = time.time() - t0
        results[h] = (t, history['val_acc'][-1])
        print(f"h={h} time={t:.2f}s val_acc={history['val_acc'][-1]:.4f}")
    return results

if __name__ == "__main__":
    experiment()
