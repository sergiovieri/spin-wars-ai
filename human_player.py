import numpy as np

from player import Player


class HumanPlayer(Player):
    def __init__(self, input_map):
        super().__init__()
        self.input_map = input_map
        self.input = np.array([0.0, 0.0])

    def get_action(self, game) -> np.ndarray:
        return self.input

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == self.input_map[0]:
            self.input[1] += 1
        elif symbol == self.input_map[1]:
            self.input[0] -= 1
        elif symbol == self.input_map[2]:
            self.input[1] -= 1
        elif symbol == self.input_map[3]:
            self.input[0] += 1

    def on_key_release(self, symbol: int, modifiers: int):
        if symbol == self.input_map[0]:
            self.input[1] -= 1
        elif symbol == self.input_map[1]:
            self.input[0] += 1
        elif symbol == self.input_map[2]:
            self.input[1] += 1
        elif symbol == self.input_map[3]:
            self.input[0] -= 1
