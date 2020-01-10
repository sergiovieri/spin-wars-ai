import numpy as np


class Player:
    def __init__(self):
        pass

    def get_action(self, game) -> np.ndarray:
        pass

    def on_key_press(self, symbol: int, modifiers: int):
        pass

    def on_key_release(self, symbol: int, modifiers: int):
        pass
