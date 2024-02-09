from enum import Enum


class StateTargetType(Enum):
    DealerPosition = "dealer_position"
    NumPlayerCards = 'num_player_cards'
    NumTableCards = 'num_table_cards'
    Opponents = 'opponents'
