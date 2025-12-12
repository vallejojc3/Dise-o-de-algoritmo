# structures/batch_queue.py
from collections import deque

class BatchQueue:
    """
    Buffer FIFO de batches. Encolado y desencolado O(1).
    """
    def __init__(self, max_batches=10):
        self.q = deque()
        self.max_batches = max_batches

    def push(self, batch):
        if len(self.q) >= self.max_batches:
            self.q.popleft()  # descarta el más viejo
        self.q.append(batch)

    def pop(self):
        if self.q:
            return self.q.popleft()
        return None

    def __len__(self):
        return len(self.q)
