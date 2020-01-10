import numpy as np

import math

from player import Player


class CenterBot(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, game) -> np.ndarray:
        best_action = None
        closest = math.inf
        return -game.p[game.turn]
