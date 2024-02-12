import os
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


class PostFlopEvaluation:
    def __init__(self, hand: Hand, table_cards: List[Card], state: GameState):
        num_players = len(state.opponent_positions) + 1
        evaluation = Evaluation(hand, table_cards, num_players)
        self.evaluations = {
            name: evaluation.weighted_evaluation(probs, debug=True) for name, probs in self.get_hand_probabilities(state).items()}

    def __str__(self):
        equity = np.array([[eval.equity, round(100 * eval.win_percent, 2)]
                          for eval in self.evaluations.values()])
        equity = equity.round(2)
        names = np.array(self.evaluations.keys())

        df = pd.DataFrame(equity, names, ["Equity", "Win %"])

        return str(df.head(len(equity)))

    def get_hand_probabilities(self, state: GameState):
        def load_probability(blinds, position):
            file_name = f"{state.game_type.get_num_players()}_{blinds}_{position.value}_{STARTING_HAND_PROBS}"
            path = os.path.join(BASE_STARTING_HAND_DIR, file_name)
            return np.load(path)

        probs = {name: [load_probability(blinds, position) for position in state.opponent_positions]
                 for blinds, name in zip([10, 30, 80], ["Low", "Med", "High"])}

        return probs
