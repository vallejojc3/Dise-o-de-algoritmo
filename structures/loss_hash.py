# structures/loss_hash.py
class LossHash:
    """
    Diccionario simple id -> loss histórico.
    """
    def __init__(self):
        self.map = {}

    def update(self, example_id, loss):
        self.map[example_id] = loss

    def topk_ids(self, k):
        return sorted(self.map.items(), key=lambda x: x[1], reverse=True)[:k]
