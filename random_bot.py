import numpy as np

from player import Player


class RandomBot(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, game) -> np.ndarray:
        return np.random.uniform(-1, 1, size=2)
