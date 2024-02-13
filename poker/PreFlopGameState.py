from enums.GameType import GameType
from enums.Hand import Hand
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState
from utils.printing import white_text, blue_text


class PreFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, blinds: int, player_cards, gto_range):
        super().__init__(game_type, game_state, blinds)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        self.gto_range = gto_range

    def str_gto(self):
        table, rfi = self.gto_range[self.hand]
        rfi_string = ""
        if rfi is not None:
            rfi_string += f"{blue_text('RFI:', bold=True)} {rfi}"

        return f"{str(table)}\n{rfi_string}"

    def str_player_cards(self):
        return f"{white_text('Player:', bold=True)} {str(self.hand)}"

    def __str__(self) -> str:
        base = super().__str__()
        return "\n".join([base, self.str_player_cards(), "\n", self.str_gto()])
