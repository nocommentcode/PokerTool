from enums.GameType import GameType
from enums.Hand import Hand
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState


class PostFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, player_cards, table_cards):
        super().__init__(game_type, game_state)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        print([str(table_card) for table_card in table_cards])
        self.table_cards = table_cards

    def __str__(self):
        string = super().__str__()
        string += "\n"
        string += f"Player Cards: {str(self.hand)}\n"
        string += f"Table Cards: {' '.join([str(card) for card in self.table_cards])}"
        return string
