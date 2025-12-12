# structures/loss_heap.py

import heapq

class LossHeap:
    """
    Heap máximo para seleccionar los ejemplos con mayor pérdida.
    """

    def __init__(self, k):
        self.k = k
        self.heap = []

    def push(self, loss, index):
        # Insertamos como negativo para simular heap máximo
        entry = (-loss, index)
        heapq.heappush(self.heap, entry)

        # Mantener tamaño k
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)

    def top_k(self):
        """Retorna los k ejemplos con mayor pérdida."""
        return [(index, -loss) for (loss, index) in self.heap]
