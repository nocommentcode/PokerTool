from enums.GameType import GameType
from poker.GameState import GameState
from utils.printing import white_text, blue_text, green_text


class BaseGameState:
    def __init__(self, game_type: GameType, game_state: GameState, blinds: int):
        self.game_type = game_type
        self.game_state = game_state
        self.game_stage = game_state.game_stage
        self.hero = game_state.player
        self.blinds = blinds

    def header(self):
        return white_text(f"------ {self.game_stage.value} (BB: {self.blinds}) -------", bold=True)

    def spacer(self):
        return "\n" * 50

    def get_pos(self):
        return f"{white_text('Hero:', bold=True)} {str(self.hero)}"

    def get_opponents(self):
        display_elipsis = len(self.game_state.opponents) > 3
        displayed_opponents = self.game_state.opponents
        if display_elipsis:
            displayed_opponents = displayed_opponents[:4]

        return f"{white_text('Opponents:', bold=True)} {', '.join(str(opponent) for opponent in displayed_opponents)}{', ...' if display_elipsis else ''}"

    def __str__(self) -> str:
        return "\n".join([self.spacer(), self.header(), self.get_pos(), self.get_opponents(), "\n"])
