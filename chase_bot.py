import copy
import math

import numpy as np

from game import Game
from player import Player


class ChaseBot(Player):
    def __init__(self):
        super().__init__()

    @staticmethod
    def simulate(steps):
        pass

    @staticmethod
    def get_angle(p0, p1):
        angle = math.acos(np.dot(p0, p1) / (np.linalg.norm(p0) * np.linalg.norm(p1)))
        # if angle > np.pi:
        #     angle = np.pi * 2 - angle

        return angle

    def get_action(self, game) -> np.ndarray:
        best_action = np.random.uniform(-1, 1, size=2)
        closest = (1e9, 1e9)

        if np.linalg.norm(game.p[0] - game.p[1]) < Game.PLAYER_RADIUS * 3:
            return game.p[1 - game.turn] - game.p[game.turn]

        if np.linalg.norm(game.p[0] - game.p[1]) > 1.0 or np.linalg.norm(game.p[game.turn]) > np.linalg.norm(
                game.p[1 - game.turn]):
            return -game.p[game.turn]

        for _ in range(16):
            action = Game.normalize(np.random.uniform(-1, 1, size=2))
            new_game = copy.deepcopy(game)
            new_game.step(action)

            for _ in range(3):
                new_game.step(new_game.last_action[new_game.turn])

            if new_game.state == Game.STATE_FINISHED and new_game.winner != game.turn:
                continue

            angle = self.get_angle(new_game.p[new_game.turn] - new_game.p[1 - new_game.turn],
                                   -new_game.p[1 - new_game.turn])

            print(angle)

            distance = np.linalg.norm(new_game.p[0] - new_game.p[1])
            # if angle < closest[0]:
            if (closest[0] > 0.1 and angle < closest[0]) or (angle < 0.1 and distance < closest[1]):
                closest = (angle, distance)
                best_action = action
        return best_action

        # collision_time = (np.linalg.norm(game.p[0] - game.p[1]) - Game.PLAYER_RADIUS * 2) / (
        #         np.linalg.norm(game.v[0]) + np.linalg.norm(game.v[1]))
        # return game.p[1 - game.turn] + game.v[1 - game.turn] * collision_time - game.p[game.turn]
