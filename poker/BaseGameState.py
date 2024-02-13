from enums.GameType import GameType
from poker.GameState import GameState
from utils.printing import white_text, blue_text, green_text


class BaseGameState:
    def __init__(self, game_type: GameType, game_state: GameState, blinds: int):
        self.game_type = game_type
        self.game_state = game_state
        self.game_stage = game_state.game_stage
        self.position = game_state.position
        self.blinds = blinds

    def header(self):
        return white_text(f"------ {self.game_stage.value} ({self.blinds}BB) -------", bold=True)

    def spacer(self):
        return "\n" * 50

    def get_pos(self):
        return f"{white_text('Position', bold=True):} {self.position.pretty_str()}"

    def get_opponents(self):
        display_elipsis = len(self.game_state.opponent_positions) > 4
        displayed_opponents = self.game_state.opponent_positions
        if display_elipsis:
            displayed_opponents = displayed_opponents[:4]

        return f"{white_text('Opponents:', bold=True)} {', '.join(position.pretty_str() for position in displayed_opponents)}{', ...' if display_elipsis else ''}"

    def __str__(self) -> str:
        return "\n".join([self.spacer(), self.header(), self.get_pos(), self.get_opponents(), "\n"])
