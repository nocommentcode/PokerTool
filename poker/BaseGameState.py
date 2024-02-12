from enums.GameType import GameType
from poker.GameState import GameState


class BaseGameState:
    def __init__(self, game_type: GameType, game_state: GameState, blinds: int):
        self.game_type = game_type
        self.game_state = game_state
        self.game_stage = game_state.game_stage
        self.position = game_state.position
        self.blinds = blinds

    def header(self):
        return f"------ {self.game_stage.value} ({len(self.game_state.opponent_positions)} opponents, {self.blinds}BB) -------"

    def spacer(self):
        return "\n" * 50

    def get_pos(self):
        return f"Position: {str(self.position)}"

    def __str__(self) -> str:
        return "\n".join([self.spacer(), self.header(), self.get_pos()])
