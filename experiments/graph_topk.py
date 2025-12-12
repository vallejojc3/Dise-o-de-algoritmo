# experiments/graph_topk.py

import matplotlib.pyplot as plt
import random
import time
from selection.topk_sort_vs_heap import topk_sort, topk_heap

def graph_topk():
    n_values = [10_000, 50_000, 100_000]
    k_values = [10, 100, 1000]

    sort_times = []
    heap_times = []
    labels = []

    for n in n_values:
        arr = [random.random() for _ in range(n)]
        for k in k_values:
            labels.append(f"{n}|k={k}")

            t0 = time.time()
            topk_sort(arr, k)
            sort_times.append(time.time() - t0)

            t0 = time.time()
            topk_heap(arr, k)
            heap_times.append(time.time() - t0)

    # Gráfica comparativa
    plt.figure(figsize=(12,6))
    plt.plot(labels, sort_times, marker='o', label="Sort O(n log n)")
    plt.plot(labels, heap_times, marker='o', label="Heap O(n log k)")
    plt.xticks(rotation=45)
    plt.title("Comparación Top-K: Sort vs Heap")
    plt.ylabel("Tiempo (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("topk_comparison.png")
    print("Imagen guardada: topk_comparison.png")

if __name__ == "__main__":
    graph_topk()
