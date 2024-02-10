from enums.Hand import Hand
from poker.GameState import GameState
from ranges.Evaluation import Evaluation


import numpy as np
import pandas as pd


class PreFlopEvaluation:
    def __init__(self, hand: Hand, state: GameState):
        self.hand = hand
        num_players = len(state.opponent_positions) + 1
        evaluation = Evaluation(self.hand, [], num_players)
        self.results = evaluation.random_evaluation()

    def get_hand_percentages(self):
        equity = np.array([eval.get_hit_percent()
                          for eval in self.results.evalutions[:-1]])
        equity = equity.round(0)
        hands = np.array([eval.name for eval in self.results.evalutions])

        indices = np.where(equity > 5)
        equity = equity.astype(int)

        df = pd.DataFrame(equity[indices], hands[indices], ["Hit %"])

        return str(df.head(len(equity)))

    def __str__(self):
        return str(self.get_hand_percentages()) + f"\nTotal Equity: {self.results.equity}" + f"\nWin %: {round(self.results.win_percent *100, 0)}"
