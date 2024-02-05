from enum import Enum


class StateTargetType(Enum):
    DealerPosition = "dealer_position"
    NumPlayerCards = 'num_player_cards'
    NumTableCards = 'num_table_cards'
    Opponent1 = 'opponent_1'
    Opponent2 = 'opponent_2'
    Opponent3 = 'opponent_3'
    Opponent4 = 'opponent_4'
    Opponent5 = 'opponent_5'
    Opponent6 = 'opponent_6'


OPPONENTS = [StateTargetType.Opponent1, StateTargetType.Opponent2,
             StateTargetType.Opponent3, StateTargetType.Opponent4, StateTargetType.Opponent5]
