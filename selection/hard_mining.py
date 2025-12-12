# selection/hard_mining.py
import heapq

class HardMiner:
    """
    Mantiene top-k ejemplos por pérdida usando min-heap.
    Cada entrada: (loss, example_id)
    """
    def __init__(self, k):
        self.k = k
        self.heap = []

    def consider(self, loss, example_id):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (loss, example_id))
        else:
            if loss > self.heap[0][0]:
                heapq.heapreplace(self.heap, (loss, example_id))

    def topk(self):
        return sorted(self.heap, reverse=True)
