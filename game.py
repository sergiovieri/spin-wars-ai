import numpy as np


class Game:
    ARENA_RADIUS = 1.0
    PLAYER_RADIUS = 0.1
    STATE_STARTED = 0
    STATE_FINISHED = 1

    ACCELERATION = 25
    FRICTION = 4

    TOUCH_SPEED = 10
    TOUCH_MULTIPLIER = 1.5

    GAME_DELTA_TIME = 0.01

    @staticmethod
    def create_p():
        while True:
            p = np.array([np.random.uniform(-1, 1, size=2), np.random.uniform(-1, 1, size=2)])
            d0 = np.linalg.norm(p[0])
            d1 = np.linalg.norm(p[1])
            d = np.linalg.norm(p[0] - p[1])
            if d0 > 0.8 or d1 > 0.8 or d < 0.2 or abs(d0 - d1) > 0.1:
                continue
            return p

    def __init__(self):
        # self.p = np.array([[-self.PLAYER_RADIUS - np.random.uniform(0, 0.01), np.random.uniform(-0.01, 0.01)],
        #                    [self.PLAYER_RADIUS + np.random.uniform(0, 0.01), np.random.uniform(-0.01, 0.01)]])
        self.p = Game.create_p()
        self.v = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.last_action = np.array([self.p[1] - self.p[0], self.p[0] - self.p[1]])
        self.turn = 0
        self.state = self.STATE_STARTED
        self.winner = None

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm != 0:
            return v / norm
        return v

    def step_internal(self, delta_time):
        v = self.v
        p = self.p
        v += self.last_action * Game.ACCELERATION * delta_time
        v -= v * Game.FRICTION * delta_time
        p += v * delta_time

        if np.linalg.norm(p[0] - p[1]) < Game.PLAYER_RADIUS * 2:
            self.resolve_collisions()

        out0 = np.linalg.norm(p[0]) > Game.ARENA_RADIUS
        out1 = np.linalg.norm(p[1]) > Game.ARENA_RADIUS
        if out0 or out1:
            self.state = self.STATE_FINISHED
            if out0 and out1:
                self.winner = 2
            elif out0:
                self.winner = 1
            else:
                self.winner = 0

    # def step(self, action, delta_time=0.01):
    #     self.turn_time -= delta_time
    #     self.last_action[self.turn] = self.normalize(action)
    #
    #     while delta_time > 0:
    #         cur = min(delta_time, Game.GAME_DELTA_TIME)
    #         self.step_internal(cur)
    #         delta_time -= cur
    #
    #     if self.turn_time < 0:
    #         self.turn = 1 - self.turn
    #         self.turn_time = Game.TURN_TIME

    def step(self, action):
        self.last_action[self.turn] = self.normalize(action)
        self.step_internal(Game.GAME_DELTA_TIME)
        self.turn = 1 - self.turn

    def resolve_collisions(self):
        v = self.v
        p = self.p
        dv = v[0] - v[1]
        dp = p[0] - p[1]
        delta = (np.dot(dv, dp) / np.dot(dp, dp)) * dp * Game.TOUCH_MULTIPLIER
        delta += -dp * Game.TOUCH_SPEED
        v[0] -= delta
        v[1] += delta
