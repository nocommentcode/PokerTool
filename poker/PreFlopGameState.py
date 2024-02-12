from enums.GameType import GameType
from enums.Hand import Hand
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState


class PreFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, blinds: int, player_cards, gto_range):
        super().__init__(game_type, game_state, blinds)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        self.gto_range = gto_range

    def str_gto(self):
        table, rfi = self.gto_range[self.hand]
        return f"RFI: {rfi}\n{str(table)}"

    def str_player_cards(self):
        return f"Player Cards: {str(self.hand)}"

    def __str__(self) -> str:
        base = super().__str__()
        return "\n".join([base, self.str_player_cards(), "\n", self.str_gto()])
