from enum import Enum


class Position(Enum):
    UTG = "utg"
    HJ = "hj"
    CO = "co"
    BTN = "btn"
    SB = "sb"
    BB = "bb"

    @staticmethod
    def from_dealer_pos(dealer_pos):
        positions = [Position.BTN, Position.SB, Position.BB,
                     Position.UTG, Position.HJ, Position.CO]
        return positions[dealer_pos]

    def __str__(self):
        return f"# {ORDERED_POSITIONS.index(self)}"

    @staticmethod
    def from_string(string):
        positions = {
            "utg": Position.UTG,
            "hj": Position.HJ,
            "co": Position.CO,
            "btn": Position.BTN,
            "sb": Position.SB,
            "bb": Position.BB}
        return positions[string]


ORDERED_POSITIONS = [Position.BTN, Position.SB, Position.BB,
                     Position.UTG, Position.HJ, Position.CO]
