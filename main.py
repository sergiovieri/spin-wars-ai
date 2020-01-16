from typing import List

import arcade
import numpy as np

from center_bot import CenterBot
from chase_bot import ChaseBot
from game import Game
from human_player import HumanPlayer
from nn_bot import NNBot
from player import Player
from random_bot import RandomBot

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800


class GameWindow(arcade.Window):
    SCALE = 400
    NEW_GAME_FREEZE = 1.0
    GAME_SPEED = 0.75

    def __init__(self, width: int, height: int, players: List[Player]):
        super().__init__(width, height)
        arcade.set_background_color(arcade.color.AMAZON)
        self.game = Game()
        self.scores = [0, 0]
        self.freeze = self.NEW_GAME_FREEZE
        self.players = players

        self.background_list = arcade.ShapeElementList()
        self.background_list.append(arcade.create_ellipse_filled(*self.translate_point(np.array([0, 0])),
                                                                 width=self.game.ARENA_RADIUS * self.SCALE,
                                                                 height=self.game.ARENA_RADIUS * self.SCALE,
                                                                 color=arcade.color.ORANGE_PEEL))

        self.p_list = [arcade.ShapeElementList() for _ in range(2)]
        self.p_list[0].append(
            arcade.create_ellipse_filled(0, 0,
                                         width=self.game.PLAYER_RADIUS * self.SCALE,
                                         height=self.game.PLAYER_RADIUS * self.SCALE,
                                         color=arcade.color.AERO_BLUE))
        self.p_list[1].append(
            arcade.create_ellipse_filled(0, 0,
                                         width=self.game.PLAYER_RADIUS * self.SCALE,
                                         height=self.game.PLAYER_RADIUS * self.SCALE,
                                         color=arcade.color.ANTIQUE_RUBY))

        self.bar_list = arcade.ShapeElementList()
        self.bar_list.append(
            arcade.create_rectangle_outline(40, 120, 40, 200, arcade.color.BLACK)
        )
        self.bar_list.append(
            arcade.create_rectangle_outline(self.width - 40, 120, 40, 200, arcade.color.BLACK)
        )

        self.skip = 0

    def translate_point(self, point: np.ndarray):
        # (-1, 1) to (0, MAX)
        return self.SCALE * point + [self.width // 2, self.height // 2]

    def draw_score(self, score, start_x):
        mid_y = 120
        if score > 0.5:
            arcade.draw_line(start_x, mid_y - 1, start_x, mid_y + int(round((score - 0.5) * 200)),
                             arcade.color.BLUE_SAPPHIRE, 40)
        elif score < 0.5:
            arcade.draw_line(start_x, mid_y - int(round((0.5 - score) * 200)), start_x, mid_y + 1,
                             arcade.color.RED_DEVIL, 40)

    def on_draw(self):
        arcade.start_render()
        self.background_list.draw()
        self.p_list[0].center_x, self.p_list[0].center_y = self.translate_point(self.game.p[0])
        self.p_list[1].center_x, self.p_list[1].center_y = self.translate_point(self.game.p[1])
        self.p_list[0].draw()
        self.p_list[1].draw()
        arcade.draw_text('{} - {}'.format(self.scores[0], self.scores[1]), 20, self.height - 20,
                         arcade.color.BLACK, font_size=36, anchor_y='top')
        self.bar_list.draw()
        self.draw_score(self.players[0].get_score(), 40)
        self.draw_score(self.players[1].get_score(), self.width - 40)

    def on_update(self, delta_time):
        if self.freeze > 0:
            self.freeze -= delta_time
            self.players[0].get_action(self.game)
            self.players[1].get_action(self.game)
            return

        self.game.step(self.players[self.game.turn].get_action(self.game))

        if self.game.state == self.game.STATE_FINISHED:
            if self.game.winner < 2:
                self.scores[self.game.winner] += 1
            self.game = Game()
            self.freeze = self.NEW_GAME_FREEZE

    def on_key_press(self, symbol: int, modifiers: int):
        for player in self.players:
            player.on_key_press(symbol, modifiers)

    def on_key_release(self, symbol: int, modifiers: int):
        for player in self.players:
            player.on_key_release(symbol, modifiers)


def main():
    input_map = [[arcade.key.W, arcade.key.A, arcade.key.S, arcade.key.D],
                 [arcade.key.UP, arcade.key.LEFT, arcade.key.DOWN, arcade.key.RIGHT]]

    game_window = GameWindow(SCREEN_WIDTH, SCREEN_HEIGHT, players=[
        # HumanPlayer(input_map=input_map[0]),
        # HumanPlayer(input_map=input_map[1]),
        # CenterBot(),
        # ChaseBot(),
        # ChaseBot(),
        # GreedyBot(),
        # RandomBot(),
        # RandomBot(),
        NNBot(),
        NNBot(),
        # ChaseBot(),
    ])
    arcade.run()


if __name__ == '__main__':
    main()
