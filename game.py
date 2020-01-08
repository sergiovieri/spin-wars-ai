import math

import arcade
import numpy as np

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800


class Game:
    ARENA_RADIUS = 1.0
    PLAYER_RADIUS = 0.1
    STATE_STARTED = 0
    STATE_FINISHED = 1

    ACCELERATION = 25
    FRICTION = 0.05

    TOUCH_SPEED = 20

    def __init__(self):
        self.p = [np.array([-self.PLAYER_RADIUS, 0.0]), np.array([self.PLAYER_RADIUS, 0.0])]
        self.v = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        self.turn = 0
        self.state = self.STATE_STARTED
        self.winner = None

    def step(self, action, delta_time=0.1):
        self.v[self.turn] += action * self.ACCELERATION * delta_time

        for i in range(2):
            self.v[i] -= self.v[i] * self.FRICTION

            self.p[i] += self.v[i] * delta_time
            if np.linalg.norm(self.p[i]) > self.ARENA_RADIUS:
                self.state = self.STATE_FINISHED
                self.winner = 1 - i

        self.resolve_collisions()
        self.turn = 1 - self.turn

    def resolve_collisions(self):
        if np.linalg.norm(self.p[0] - self.p[1]) < self.PLAYER_RADIUS * 2:
            self.v = [self.calculate_collision(self.v[0], self.v[1], self.p[0], self.p[1]),
                      self.calculate_collision(self.v[1], self.v[0], self.p[1], self.p[0])]

            direction = (self.p[1] - self.p[0]) * self.TOUCH_SPEED
            self.v[0] -= direction
            self.v[1] += direction

    @staticmethod
    def calculate_collision(v1, v2, x1, x2):
        return v1 - (np.dot(v1 - v2, x1 - x2) / np.dot(x1 - x2, x1 - x2)) * (x1 - x2)


class GameWindow(arcade.Window):
    INPUT_MAP = [[arcade.key.W, arcade.key.A, arcade.key.S, arcade.key.D],
                 [arcade.key.UP, arcade.key.LEFT, arcade.key.DOWN, arcade.key.RIGHT]]

    SCALE = 400
    NEW_GAME_FREEZE = 1.0
    GAME_SPEED = 0.75

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        arcade.set_background_color(arcade.color.AMAZON)
        self.game = Game()
        self.input = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        self.scores = [0, 0]
        self.freeze = 0

    def setup(self):
        pass

    def translate_point(self, point):
        # (-1, 1) to (0, MAX)
        return self.width // 2 + point[0] * self.SCALE, self.height // 2 + point[1] * self.SCALE

    def on_draw(self):
        arcade.start_render()
        arcade.draw_circle_filled(*self.translate_point((0, 0)), self.game.ARENA_RADIUS * self.SCALE,
                                  arcade.color.ORANGE_PEEL)
        arcade.draw_circle_filled(*self.translate_point(self.game.p[0]), self.game.PLAYER_RADIUS * self.SCALE,
                                  arcade.color.AERO_BLUE)
        arcade.draw_circle_filled(*self.translate_point(self.game.p[1]), self.game.PLAYER_RADIUS * self.SCALE,
                                  arcade.color.ANTIQUE_RUBY)
        arcade.draw_text('{} - {}'.format(self.scores[0], self.scores[1]), 10, 10, arcade.color.BLACK_OLIVE,
                         font_size=20)

    @staticmethod
    def normalize_input(v):
        norm = np.linalg.norm(v)
        if norm != 0:
            return v / norm
        return v

    def update(self, delta_time):
        if self.freeze > 0:
            self.freeze -= delta_time
            return
        turn = self.game.turn
        self.game.step(self.normalize_input(self.input[turn]), delta_time * self.GAME_SPEED)
        if self.game.state == self.game.STATE_FINISHED:
            self.scores[self.game.winner] += 1
            self.game = Game()
            self.freeze = self.NEW_GAME_FREEZE

    def on_key_press(self, symbol: int, modifiers: int):
        for player in range(2):
            if symbol == self.INPUT_MAP[player][0]:
                self.input[player][1] += 1
            elif symbol == self.INPUT_MAP[player][1]:
                self.input[player][0] -= 1
            elif symbol == self.INPUT_MAP[player][2]:
                self.input[player][1] -= 1
            elif symbol == self.INPUT_MAP[player][3]:
                self.input[player][0] += 1

    def on_key_release(self, symbol: int, modifiers: int):
        for player in range(2):
            if symbol == self.INPUT_MAP[player][0]:
                self.input[player][1] -= 1
            elif symbol == self.INPUT_MAP[player][1]:
                self.input[player][0] += 1
            elif symbol == self.INPUT_MAP[player][2]:
                self.input[player][1] += 1
            elif symbol == self.INPUT_MAP[player][3]:
                self.input[player][0] -= 1


def main():
    game_window = GameWindow(SCREEN_WIDTH, SCREEN_HEIGHT)
    game_window.setup()
    arcade.run()


if __name__ == '__main__':
    main()
