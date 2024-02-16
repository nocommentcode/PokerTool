from typing import List
import numpy as np
from enums.GameStage import GameStage
from enums.Position import Position
from poker.Player import Player
from utils.number_extraction import get_opponent_stack_sizes


class GameState:
    def __init__(self, game_type, blinds, player_cards, table_cards, dealer_pos, opponents, screenshot):
        self.game_type = game_type
        self.blinds = blinds
        self.player_card_count = player_cards
        self.table_card_count = table_cards
        self.dealer_pos = dealer_pos
        self.game_stage = self.get_game_stage(player_cards, table_cards)
        self.opponent_indexes = np.where(opponents)[0]
        self.player_position = self.get_position(self.dealer_pos)
        self.screenshot = screenshot

    def get_position(self, dealer_pos) -> Position:
        return Position.from_dealer_pos_idx(dealer_pos, self.game_type)

    def get_game_stage(self, player_cards, table_cards) -> GameStage:
        if player_cards == 0:
            return GameStage.FOLDED

        if table_cards == 0:
            return GameStage.PREFLOP

        if table_cards == 3:
            return GameStage.FLOP

        if table_cards == 4:
            return GameStage.Turn

        if table_cards == 5:
            return GameStage.River

    def __eq__(self, other):
        if type(other) != GameState:
            return False

        if self.player_position != other.player_position:
            return False

        if self.game_stage != other.game_stage:
            return False

        if np.all(self.opponent_indexes != other.opponent_indexes):
            return False

        return True

    def parse_players(self, prev_state: "GameState", ):
        if self.game_stage == GameStage.PREFLOP or prev_state is None:
            self.player = Player(self.game_type, self.blinds,
                                 0, self.player_position, self.screenshot)
            self.opponents = [Player(self.game_type, self.blinds, i+1, self.player_position.get_relative_pos(
                i, self.game_type), self.screenshot) for i in self.opponent_indexes]

        else:
            self.player = prev_state.player
            remaining_opponents = [self.player_position.get_relative_pos(
                i, self.game_type) for i in self.opponent_indexes]
            self.opponents = [
                opponent for opponent in prev_state.opponents if opponent.position in remaining_opponents]
