from enums.GameType import GameType
from enums.Hand import Hand
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState
from ranges.PostFlopEvaluation import PostFlopEvaluation
from utils.printing import white_text


class PostFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, blinds: int, player_cards, table_cards):
        super().__init__(game_type, game_state, blinds)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        self.table_cards = table_cards
        self.evaluation = PostFlopEvaluation(
            self.hand, table_cards, game_state)

    def str_player_cards(self):
        return f"{white_text('Player:', bold=True)} {str(self.hand)}"

    def str_table_cards(self):
        return f"{white_text('Table:', bold=True)} {' '.join([str(card) for card in self.table_cards])}"

    def __str__(self) -> str:
        base = super().__str__()
        return "\n".join([base, self.str_player_cards(), "\n", self.str_table_cards(), "\n", str(self.evaluation)])
