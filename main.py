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

    def translate_point(self, point: np.ndarray):
        # (-1, 1) to (0, MAX)
        return self.SCALE * point + [self.width // 2, self.height // 2]

    def on_draw(self):
        arcade.start_render()
        self.background_list.draw()
        self.p_list[0].center_x, self.p_list[0].center_y = self.translate_point(self.game.p[0])
        self.p_list[1].center_x, self.p_list[1].center_y = self.translate_point(self.game.p[1])
        self.p_list[0].draw()
        self.p_list[1].draw()
        # arcade.draw_circle_filled(*self.translate_point(np.array([0, 0])), self.game.ARENA_RADIUS * self.SCALE,
        #                           arcade.color.ORANGE_PEEL)
        # arcade.draw_circle_filled(*self.translate_point(self.game.p[0]), self.game.PLAYER_RADIUS * self.SCALE,
        #                           arcade.color.AERO_BLUE)
        # arcade.draw_circle_filled(*self.translate_point(self.game.p[1]), self.game.PLAYER_RADIUS * self.SCALE,
        #                           arcade.color.ANTIQUE_RUBY)
        arcade.draw_text('{} - {}'.format(self.scores[0], self.scores[1]), 20, self.height - 20,
                         arcade.color.BLACK, font_size=36, anchor_y='top', font_name='')

    def on_update(self, delta_time):
        print(delta_time)
        if self.freeze > 0:
            self.freeze -= delta_time
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
