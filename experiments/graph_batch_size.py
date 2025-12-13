# experiments/graph_batch_size.py

import matplotlib.pyplot as plt
from experiments.exp_batch_size import experiment


def graph_batch_size():
    results = experiment()
    batch_sizes = list(results.keys())
    times = [results[b][0] for b in batch_sizes]
    accuracies = [results[b][1] for b in batch_sizes]

    # ---- Gráfica de Tiempo ----
    plt.figure(figsize=(8,5))
    plt.plot(batch_sizes, times, marker='o')
    plt.title("Tiempo por Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Tiempo (s)")
    plt.grid(True)
    plt.savefig("graphs/batch_size_time.png")
    print("Imagen guardada: batch_size_time.png")

    # ---- Gráfica de Accuracy ----
    plt.figure(figsize=(8,5))
    plt.plot(batch_sizes, accuracies, marker='o', color='orange')
    plt.title("Accuracy por Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("graphs/batch_size_accuracy.png")
    print("Imagen guardada: batch_size_accuracy.png")

if __name__ == "__main__":
    graph_batch_size()
