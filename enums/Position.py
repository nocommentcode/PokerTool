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

    def get_relative_pos(self, index):
        positions = [pos.value for pos in Position]
        my_index = positions.index(self.value)
        their_index = (my_index + index) % len(positions)
        return positions[their_index]

    @staticmethod
    def from_dealer_pos_idx(dealer_pos):
        positions = [Position.BTN, Position.CO, Position.HJ,
                     Position.UTG, Position.BB, Position.SB]
        return positions[dealer_pos]

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
