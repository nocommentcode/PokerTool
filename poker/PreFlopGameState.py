import pandas as pd
from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.Hand import Hand
from enums.Position import Position
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState
from ranges.evalution import Evaluation


class PreFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, player_cards, charts):
        super().__init__(game_type, game_state)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        self.charts = charts
        self.evaluation = self.evaluate_hand()

    def get_gto_ranges(self):
        actions = [chart[self.hand] for chart in self.charts.values()]
        names = [actions for actions in self.charts.keys()]
        df = pd.DataFrame(actions, names, ["GTO"])
        return df.head(len(actions))

    def evaluate_hand(self):
        num_players = len(self.game_state.opponent_positions) + 1
        eval = Evaluation(self.hand, [], num_players)
        return eval.run_evaluation()

    def str_player_cards(self):
        return f"Player Cards: {str(self.hand)}"

    def str_evaluation(self):
        return str(self.evaluation.get_hand_percentages()) + f"\nTotal Equity: {self.evaluation.equity}"

    def __str__(self) -> str:
        base = super().__str__()
        return "\n".join([base, self.str_player_cards(), "\n", str(self.get_gto_ranges()), "\n", self.str_evaluation()])
