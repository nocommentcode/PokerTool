from enums.GameType import GameType
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState


class FoldedGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, blinds: int):
        super().__init__(game_type, game_state, blinds)
