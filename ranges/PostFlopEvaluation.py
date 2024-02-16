import os
import time
from enums.Card import Card
from enums.GameType import GameType
from enums.Hand import Hand
from enums.Looseness import Looseness
from enums.Position import Position
from poker.GameState import GameState
from ranges import BASE_STARTING_HAND_DIR, STARTING_HAND_PROBS
from ranges.Evaluation import Evaluation


import numpy as np
import pandas as pd


from typing import List

from utils.PrettyTable import PrettyTable


class PostFlopEvaluation:
    def __init__(self, hand: Hand, table_cards: List[Card], state: GameState):
        num_players = len(state.opponents) + 1
        evaluation = Evaluation(hand, table_cards, num_players)
        self.evaluations = {
            name: evaluation.weighted_evaluation(probs) for name, probs in self.get_hand_probabilities(state).items()}

    def __str__(self):
        equity = np.array([[eval.equity, round(100 * eval.win_percent, 2)]
                          for eval in self.evaluations.values()])
        equity = equity.round(2)

        table = PrettyTable("Equity Eval", "blue", 3)
        table.add_row_names(self.evaluations.keys())
        table.add_data(equity, ["Equity", "Win %"])
        return str(table)

    def get_hand_probabilities(self, state: GameState):
        def get_gto_probability(blinds, position):
            file_name = f"{state.game_type.get_num_players()}_{blinds}_{position.value}_{STARTING_HAND_PROBS}"
            path = os.path.join(BASE_STARTING_HAND_DIR, file_name)
            return np.load(path)

        def get_range_probabilities(range):
            num_hands = 52*51
            probs = np.zeros((num_hands,))

            if range == 0:
                range = int(num_hands * 0.2)
            else:
                range = int(num_hands * range / 100)
            probs[:range] = 1

            return probs

        probs = {
            "GTO": [get_gto_probability(player.stack_size.value, player.position) for player in state.opponents],
            "Range": [get_range_probabilities(player.range) for player in state.opponents]
        }

        return probs
