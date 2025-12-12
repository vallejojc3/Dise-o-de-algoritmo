# experiments/exp_ra3.py

import time
import numpy as np

from data.load_mnist import load_mnist_subset
from mlp.mlp_from_scratch import MLP
from experiments.exp_batch_size import train

from structures.batch_queue import BatchQueue
from structures.loss_dict import LossDict
from structures.loss_heap import LossHeap


def train_with_structures(X, y, epochs=5, batch=64):
    N = X.shape[0]

    # CORRECTO: redondeo hacia arriba
    batches = (N + batch - 1) // batch

    losses = LossDict()
    heap = LossHeap(k=10)

    model = MLP(784, 64, 10)

    for e in range(epochs):

        # RECARGAR LA COLA EN CADA ÉPOCA
        queue = BatchQueue(batches)
        for b in range(batches):
            lo = b * batch
            hi = min((b + 1) * batch, N)  # CORRECTO: evita overflow
            queue.enqueue((lo, hi))

        for batch_id in range(batches):

            lo, hi = queue.dequeue()
            Xb, yb = X[lo:hi], y[lo:hi]

            probs, cache = model.forward(Xb)
            loss, grads = model.compute_loss_and_grads(cache, yb)

            # almacenar pérdida
            losses.update(batch_id, float(loss))

            # almacenar ejemplos duros
            for i in range(hi - lo):
                heap.push(float(loss), lo + i)

            model.update_params(grads)

    return model, losses, heap



def experiment():
    X_train, y_train, _, _ = load_mnist_subset(2000, 500)

    # normal
    t0 = time.time()
    model_norm = MLP(784, 64, 10)
    train(model_norm, X_train, y_train, X_train, y_train, epochs=5, verbose=False)
    t_norm = time.time() - t0

    # con estructuras
    t0 = time.time()
    train_with_structures(X_train, y_train)
    t_struct = time.time() - t0

    print(f"\nTiempo sin estructuras:  {t_norm:.3f} s")
    print(f"Tiempo con estructuras:  {t_struct:.3f} s")

    return t_norm, t_struct


if __name__ == "__main__":
    experiment()
