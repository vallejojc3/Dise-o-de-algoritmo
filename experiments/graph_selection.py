# experiments/graph_selection.py

import matplotlib.pyplot as plt
from experiments.exp_selection import experiment

def graph_selection():
    results = experiment()

    labels = list(results.keys())
    sort_times = [results[key][0] for key in labels]
    heap_times = [results[key][1] for key in labels]
    qs_times   = [results[key][2] for key in labels]

    plt.figure(figsize=(12, 6))
    plt.plot(labels, sort_times, marker='o', label="Sort O(n log n)")
    plt.plot(labels, heap_times, marker='o', label="Heap O(n log k)")
    plt.plot(labels, qs_times, marker='o', label="Quickselect O(n)")
    plt.xticks(rotation=45)
    plt.ylabel("Tiempo (s)")
    plt.title("Comparación de métodos Top-K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("topk_methods_comparison.png")
    print("Imagen: topk_methods_comparison.png guardada.")

if __name__ == "__main__":
    graph_selection()
