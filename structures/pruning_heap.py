# structures/pruning_heap.py
import heapq

class PruningHeap:
    """
    Heap para candidatos a poda (por magnitud).
    Guarda (abs_metric, param_ref) y permite obtener los menores.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.heap = []

    def consider(self, metric, identifier):
        # metric: menor => candidato a poda
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, (metric, identifier))
        else:
            if metric > self.heap[0][0]:
                heapq.heapreplace(self.heap, (metric, identifier))

    def candidates(self):
        return sorted(self.heap)
