from typing import List

from enums.Card import Card

from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.Hand import Hand
from enums.Position import Position


class GameStated:
    def __init__(self, num_players: int, dealer_pos: int, player_cards: List[Card], table_cards: List[Card]):
        self.num_players = num_players
        self.dealer_pos = dealer_pos
        self.player_cards = player_cards
        self.table_cards = table_cards

        self.game_stage = self.get_game_stage()

    def get_game_stage(self):
        if len(self.player_cards) == 0:
            return GameStage.FOLDED

        if len(self.table_cards) == 0:
            return GameStage.PREFLOP

        if len(self.table_cards) == 3:
            return GameStage.FLOP

        if len(self.table_cards) == 4:
            return GameStage.Turn

        if len(self.table_cards) == 5:
            return GameStage.River

    def __str__(self) -> str:
        string = f"------ {self.game_stage} ({self.num_players} players) -------\n"
        string += f"Position: {self.dealer_pos}\n\n"

        string += "Player Cards: "
        for card in self.player_cards:
            string += str(card) + "  "
        string += "\n"

        string += "Table Cards: "
        for card in self.table_cards:
            string += str(card) + "  "
        string += "\n"

        return string

    def __eq__(self, other) -> bool:
        if type(other) != GameState:
            return False

        if self.game_stage != other.game_stage:
            return False

        if self.num_players != other.num_players:
            return False

        if self.dealer_pos != other.dealer_pos:
            return False

        for me, them in zip(self.player_cards, other.player_cards):
            if me != them:
                return False

        for me, them in zip(self.table_cards, other.table_cards):
            if me != them:
                return False

        return True


class GameState:
    def __init__(self, game_type: GameType, position: Position, game_stage: GameStage):
        self.game_type = game_type
        self.game_stage = game_stage
        self.position = position

    def __str__(self) -> str:
        string = f"------ {self.game_stage.value} ({self.game_type.value}) -------\n"
        string += f"Position {str(self.position)}\n\n"
        return string


class FoldedGameState(GameState):
    def __init__(self, game_type: GameType, position: Position):
        super().__init__(game_type, position, GameStage.FOLDED)


class PreFlopGameState(GameState):
    def __init__(self, game_type: GameType, position: Position, player_cards, charts):
        print(charts)
        super().__init__(game_type, position, GameStage.PREFLOP)
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
