from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.Hand import Hand
from enums.Position import Position
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState


class PreFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, player_cards, charts):
        super().__init__(game_type, game_state)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        self.charts = charts

    def get_gto_ranges(self):
        return {action: value for action, value in self.charts.items()}

    def __str__(self) -> str:
        string = super().__str__()
        string += "\n"
        string += f"Player Cards: {str(self.hand)}"
        string += "\n---- GTO Range -----\n"
        for action, value in reversed(self.get_gto_ranges().items()):
            string += f"{str(action)}: {value[self.hand]}\n"

        return string
