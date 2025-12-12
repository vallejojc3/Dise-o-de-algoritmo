# selection/topk_sort_vs_heap.py
import heapq
import random
import time

def topk_sort(arr, k):
    return sorted(arr, reverse=True)[:k]

def topk_heap(arr, k):
    if k <= 0:
        return []
    heap = arr[:k]
    heapq.heapify(heap)
    for x in arr[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)

def benchmark():
    for n in [10_000, 50_000, 100_000]:
        arr = [random.random() for _ in range(n)]
        for k in [10, 100, 1000]:
            t0 = time.time()
            topk_sort(arr, k)
            t_sort = time.time() - t0
            t0 = time.time()
            topk_heap(arr, k)
            t_heap = time.time() - t0
            print(f"n={n}, k={k}: sort={t_sort:.4f}s, heap={t_heap:.4f}s")

if __name__ == "__main__":
    benchmark()
