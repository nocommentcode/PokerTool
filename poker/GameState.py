from enums.GameStage import GameStage
from enums.Position import Position


class GameState:
    def __init__(self, player_cards, table_cards, dealer_pos, num_opponents):
        self.player_card_count = player_cards
        self.table_card_count = table_cards
        self.dealer_pos = dealer_pos
        self.num_opponents = num_opponents
        self.position = self.get_position(self.dealer_pos)
        self.game_stage = self.get_game_stage(player_cards, table_cards)

    def get_position(self, dealer_pos) -> Position:
        positions = [Position.BTN, Position.CO, Position.HJ,
                     Position.UTG, Position.BB, Position.SB]
        return positions[dealer_pos]

    def get_game_stage(self, player_cards, table_cards) -> GameStage:
        if player_cards == 0:
            return GameStage.FOLDED

        if table_cards == 0:
            return GameStage.PREFLOP

        if table_cards == 3:
            return GameStage.FLOP

        if table_cards == 4:
            return GameStage.Turn

        if table_cards == 5:
            return GameStage.River

    def __eq__(self, other):
        if type(other) != GameState:
            return False

        if self.position != other.position:
            return False

        if self.game_stage != other.game_stage:
            return False

        if self.num_opponents != other.num_opponents:
            return False

        return True
