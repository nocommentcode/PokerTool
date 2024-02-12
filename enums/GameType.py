from enum import Enum


class GameType(Enum):
    SixPlayer = "6_player"
    NinePlayer = "9_player"
    EightPlayer = "8_player"

    def get_num_players(self):
        if self == GameType.SixPlayer:
            return 6
        if self == GameType.EightPlayer:
            return 8

        if self == GameType.NinePlayer:
            return 9
