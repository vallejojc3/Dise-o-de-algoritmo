# structures/loss_dict.py

class LossDict:
    """
    Mapa hash para almacenar pérdidas por batch.
    acceso: O(1)
    """

    def __init__(self):
        self.storage = {}

    def update(self, batch_id, loss):
        self.storage[batch_id] = loss

    def get(self, batch_id):
        return self.storage.get(batch_id, None)

    def max_loss_batch(self):
        """Retorna el batch con mayor pérdida."""
        if not self.storage:
            return None
        # O(n) pero n = batches en una época (pequeño)
        return max(self.storage, key=lambda k: self.storage[k])
