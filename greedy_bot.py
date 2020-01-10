import copy

import numpy as np

from game import Game
from player import Player


class GreedyBot(Player):
    def __init__(self):
        super().__init__()

    @staticmethod
    def simulate(steps):
        pass

    def get_action(self, game) -> np.ndarray:
        # best_action = None
        # closest = 1e9

        # if np.random.rand() < 0.1:
        #     return np.random.rand(2) * 2 - 1

        # for _ in range(16):
        #     action = np.random.rand(2) * 2 - 1
        #     p = np.copy(game.p)
        #     v = np.copy(game.v)
        #     turn = game.turn
        #     v[turn] += Game.normalize(action)
        #     # p += v * collision_time
        #     p += v * 0.1
        #     # current = np.linalg.norm(p[0] - p[1])
        #     current = np.linalg.norm(p[turn]) - np.linalg.norm(p[1 - turn])
        #     if current < closest:
        #         closest = current
        #         best_action = action
        #
        # return best_action

        # return game.p[1 - game.turn] + game.v[1 - game.turn] * collision_time - game.p[game.turn]

        turn = game.turn
        best_action = np.random.rand(2) * 2 - 1
        fastest = 1e9

        for _ in range(32):
            new_game = copy.deepcopy(game)
            action = np.random.rand(2) * 2 - 1
            current = 0
            new_game.step(np.zeros(2), delta_time=0.02)
            new_game.step(new_game.p[1 - new_game.turn] - new_game.p[new_game.turn], delta_time=0.02)
            new_game.step(action, delta_time=0.02)
            while new_game.state != Game.STATE_FINISHED and current < 100:
                new_game.step(new_game.p[1 - new_game.turn] - new_game.p[new_game.turn], delta_time=0.02)
                current += 1

            if new_game.state != Game.STATE_FINISHED or new_game.winner != turn:
                continue

            return action

            if current < fastest:
                fastest = current
                best_action = action

        print('fastest', fastest, best_action)

        return best_action
