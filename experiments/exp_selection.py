# experiments/exp_selection.py

import random
import time
from selection.quickselect import quickselect
from selection.topk_sort_vs_heap import topk_sort, topk_heap

def experiment():
    n_values = [10_000, 50_000, 100_000]
    k_values = [10, 100, 1000]

    results = {}

    for n in n_values:
        arr = [random.random() for _ in range(n)]
        for k in k_values:
            key = f"n={n}, k={k}"
            print(f"Evaluando {key}...")

            # Sort O(n log n)
            t0 = time.time()
            topk_sort(arr, k)
            t_sort = time.time() - t0

            # Heap O(n log k)
            t0 = time.time()
            topk_heap(arr, k)
            t_heap = time.time() - t0

            # Quickselect O(n)
            t0 = time.time()
            quickselect(arr, k)
            t_qs = time.time() - t0

            results[key] = (t_sort, t_heap, t_qs)

            print(f" sort={t_sort:.4f}s | heap={t_heap:.4f}s | quickselect={t_qs:.4f}s")

    return results

if __name__ == "__main__":
    experiment()
