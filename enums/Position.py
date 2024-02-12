from enum import Enum

from enums.GameType import GameType


class Position(Enum):
    UTG = "utg"
    UTG1 = "utg_1"
    UTG2 = "utg_2"
    LJ = "lj"
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
            Position.UTG1: "Under the gun + 1",
            Position.UTG2: "Under the gun + 2",
            Position.LJ: "Low jack",
            Position.HJ: "High jack",
            Position.CO: "Cutoff",
            Position.BTN: "Button",
        }
        return position_names[self]

    def get_relative_pos(self, index, game_type):
        positions = GAME_TYPE_POSITIONS[game_type]
        my_index = positions.index(self)
        their_index = (my_index + index) % len(positions)
        return positions[their_index]

    @staticmethod
    def from_dealer_pos_idx(dealer_pos, game_type):
        index = (game_type.get_num_players() -
                 dealer_pos) % game_type.get_num_players()
        return GAME_TYPE_POSITIONS[game_type][index]

    @staticmethod
    def from_string(string):
        positions = {
            "utg": Position.UTG,
            "utg_1": Position.UTG1,
            "utg_2": Position.UTG2,
            "lj": Position.LJ,
            "hj": Position.HJ,
            "co": Position.CO,
            "btn": Position.BTN,
            "sb": Position.SB,
            "bb": Position.BB}
        return positions[string]


GAME_TYPE_POSITIONS = {
    GameType.SixPlayer: [Position.BTN,
                         Position.SB,
                         Position.BB,
                         Position.UTG,
                         Position.HJ,
                         Position.CO],

    GameType.EightPlayer: [Position.BTN,
                           Position.SB,
                           Position.BB,
                           Position.UTG,
                           Position.UTG1,
                           Position.LJ,
                           Position.HJ,
                           Position.CO],

    GameType.NinePlayer: [Position.BTN,
                          Position.SB,
                          Position.BB,
                          Position.UTG,
                          Position.UTG1,
                          Position.UTG2,
                          Position.LJ,
                          Position.HJ,
                          Position.CO]
}
