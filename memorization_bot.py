import math

import numpy as np

from game import Game
from player import Player


class MemorizationBot(Player):
    def __init__(self):
        super().__init__()


def pick_action(game):
    return np.random.uniform(-1, 1, size=2)


def discretize(x, num, lo, hi):
    if x <= lo:
        print('min')
        return 0
    if x >= hi:
        print('max')
        return num - 1
    x -= lo
    x /= (hi - lo)
    return round(x * num)


def get_info(p, v):
    dp0 = np.linalg.norm(p[0])
    dp1 = np.linalg.norm(p[1])
    dp01 = np.linalg.norm(p[0] - p[1])
    angle = math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0])
    return (discretize(dp0, 10, 0, 2),
            discretize(dp1, 10, 0, 2),
            discretize(dp01, 10, 0, 2),
            discretize(angle, 10, 0, np.pi * 2))


def main():
    # |p0|, |p1|, |p0-p1|, angle(north-p0-p1)
    memory = np.zeros((10, 10, 10, 10, 2), dtype=np.int)
    filename = 'memory.npy'

    try:
        memory = np.load(filename)
    except IOError as e:
        print('IOError', e)

    last_saved = 0
    while True:
        game = Game()

        history = []

        while game.state != Game.STATE_FINISHED:
            history.append((np.copy(game.p), np.copy(game.v), game.turn))
            game.step(pick_action(game))

        if game.winner > 1:
            print('DRAW')
            continue

        print(len(history))

        for h in history:
            p, v, turn = h
            if turn == 1:
                p[0], p[1] = p[1], p[0]
                v[0], v[1] = v[1], v[0]

            idx = get_info(p, v)
            print(idx)
            h[idx][1] += 1
            if turn == game.winner:
                h[idx][0] += 1

        last_saved += 1
        if last_saved > 100:
            last_saved = 0
            np.save(filename, memory)


if __name__ == '__main__':
    main()
