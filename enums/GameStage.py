from enum import Enum


class GameStage(Enum):
    FOLDED = "Folded"
    PREFLOP = "Pre-Flop"
    FLOP = "Flop"
    Turn = "Turn"
    River = "River"
