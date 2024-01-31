from enum import Enum


class TargetType(Enum):
    Num_players = "num_players"
    Dealer_pos = "dealer_pos"

    Player_card_1 = "player_card_1"
    Player_card_2 = "player_card_2"

    Flop_card_1 = "flop_card_1"
    Flop_card_2 = "flop_card_2"
    Flop_card_3 = "flop_card_3"

    Turn_card = "turn_card"
    River_card = "river_card"


CARD_TARGETS = [TargetType.Player_card_1,
                TargetType.Player_card_2,
                TargetType.Flop_card_1,
                TargetType.Flop_card_2,
                TargetType.Flop_card_3,
                TargetType.Turn_card,
                TargetType.River_card]

INT_TARGETS = [TargetType.Num_players,
               TargetType.Dealer_pos]
