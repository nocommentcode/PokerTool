import pandas as pd
from enums.GameType import GameType
from enums.Hand import Hand
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState
from ranges.PreFlopEvaluation import PreFlopEvaluation


class PreFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, player_cards, charts):
        super().__init__(game_type, game_state)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        self.charts = charts
        self.evaluation = PreFlopEvaluation(self.hand, game_state)

    def get_gto_ranges(self):
        actions = [chart[self.hand] for chart in self.charts.values()]
        names = [actions for actions in self.charts.keys()]
        df = pd.DataFrame(actions, names, ["GTO"])
        return df.head(len(actions))

    def str_player_cards(self):
        return f"Player Cards: {str(self.hand)}"

    def __str__(self) -> str:
        base = super().__str__()
        return "\n".join([base, self.str_player_cards(), "\n", str(self.get_gto_ranges()), "\n", str(self.evaluation)])
