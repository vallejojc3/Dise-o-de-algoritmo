# mlp/mlp_from_scratch.py
import numpy as np

def one_hot(y, C):
    Y = np.zeros((y.size, C))
    Y[np.arange(y.size), y] = 1
    return Y

class MLP:
    """
    MLP con 1 capa oculta (ReLU) + softmax de salida.
    W1: (D, h), W2: (h, C)
    """
    def __init__(self, D, h, C, weight_scale=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, weight_scale, (D, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, weight_scale, (h, C))
        self.b2 = np.zeros(C)

    def forward(self, X):
        # X: (B, D)
        Z1 = X @ self.W1 + self.b1  # (B, h)
        A1 = np.maximum(Z1, 0)      # ReLU
        Z2 = A1 @ self.W2 + self.b2 # (B, C)
        # softmax estable
        exp = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        cache = (X, Z1, A1, Z2, probs)
        return probs, cache

    def compute_loss_and_grads(self, cache, y, reg=0.0):
        X, Z1, A1, Z2, probs = cache
        B = X.shape[0]
        C = probs.shape[1]
        Y = one_hot(y, C)
        data_loss = -np.sum(Y * np.log(probs + 1e-12)) / B
        reg_loss = 0.5 * reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        loss = data_loss + reg_loss

        dZ2 = (probs - Y) / B             # (B, C)
        dW2 = A1.T @ dZ2 + reg * self.W2  # (h, C)
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T             # (B, h)
        dZ1 = dA1 * (Z1 > 0)              # ReLU backprop
        dW1 = X.T @ dZ1 + reg * self.W1   # (D, h)
        db1 = np.sum(dZ1, axis=0)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return loss, grads

    def update_params(self, grads, lr=1e-3):
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)
