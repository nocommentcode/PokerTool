from enums.Card import Card
from enums.Hand import Hand
from enums.Looseness import Looseness
from enums.Position import Position
from poker.GameState import GameState
from ranges.Evaluation import Evaluation


import numpy as np
import pandas as pd


from typing import List


class PostFlopEvaluation:

    strengths = {
        Position.SB: 250,
        Position.BB: 300,
        Position.UTG: 350,
        Position.HJ: 400,
        Position.CO: 450,
        Position.BTN: 500,
    }

    def __init__(self, hand: Hand, table_cards: List[Card], state: GameState):
        num_players = len(state.opponent_positions) + 1
        evaluation = Evaluation(hand, table_cards, num_players)
        self.evaluations = {
            lossness: evaluation.weighted_evaluation([self.strengths[pos] for pos in state.opponent_positions], lossness.get_factor()) for lossness in Looseness}

    def __str__(self):
        equity = np.array([[eval.equity, round(100 * eval.win_percent, 2)]
                          for eval in self.evaluations.values()])
        equity = equity.round(2)
        names = np.array([type.value for type in self.evaluations.keys()])

        df = pd.DataFrame(equity, names, ["Equity", "Win %"])

        return str(df.head(len(equity)))
