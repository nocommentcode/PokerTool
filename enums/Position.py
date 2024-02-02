from enum import Enum


class Position(Enum):
    UTG = "utg"
    HJ = "hj"
    CO = "co"
    BTN = "btn"
    SB = "sb"
    BB = "bb"

    def __str__(self):
        position_names = {
            Position.BB: "Big-blind",
            Position.SB: "Small-blind",
            Position.UTG: "Under the gun",
            Position.HJ: "High jack",
            Position.CO: "Cutoff",
            Position.BTN: "Button",
        }
        return position_names[self]

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
