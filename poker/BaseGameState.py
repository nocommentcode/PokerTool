from enums.GameType import GameType
from poker.GameState import GameState


class BaseGameState:
    def __init__(self, game_type: GameType, game_state: GameState):
        self.game_type = game_type
        self.game_state = game_state
        self.game_stage = game_state.game_stage
        self.position = game_state.position

    def __str__(self) -> str:
        string = f"------ {self.game_stage.value} ({len(self.game_state.opponent_positions)} in hand) -------\n"
        string += f"Position: {str(self.position)}\n\n"
        return string
