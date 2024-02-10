from enum import Enum

from enums.Position import Position


class OpponentAction(Enum):
    RFI = "rfi"
    RAISE = "raise"
    ALL_IN = "all_in"
    THREE_BET = "three_bet"
    THREE_BET_ALL_IN = "three_bet_all_in"
    FOUR_BET_ALL_IN = "four_bet_all_in"

    def __str__(self):
        if self == OpponentAction.RFI:
            return "RFI"

        if self == OpponentAction.RAISE:
            return "Raise"

        if self == OpponentAction.ALL_IN:
            return "Allin"

        if self == OpponentAction.THREE_BET:
            return "3Bet"

        if self == OpponentAction.THREE_BET_ALL_IN:
            return "3Allin"

        if self == OpponentAction.FOUR_BET_ALL_IN:
            return "4Allin"

    def __lt__(self, other):
        if type(other) != OpponentAction:
            return False

        ordered = [OpponentAction.RFI, OpponentAction.RAISE, OpponentAction.ALL_IN, OpponentAction.THREE_BET,
                   OpponentAction.THREE_BET_ALL_IN, OpponentAction.FOUR_BET_ALL_IN]

        my_index = ordered.index(self)
        other_index = ordered.index(other)

        return my_index < other_index
