from enum import Enum


class PokerTargetType(Enum):
    PlayerCard1 = "player_card_1"
    PlayerCard2 = "player_card_2"

    FlopCard1 = "flop_card_1"
    FlopCard2 = "flop_card_2"
    FlopCard3 = "flop_card_3"

    TurnCard = "turn_card"
    RiverCard = "river_card"


PLAYER_CARDS = [PokerTargetType.PlayerCard1, PokerTargetType.PlayerCard2]
FLOP_CARDS = [PokerTargetType.FlopCard1,
              PokerTargetType.FlopCard2, PokerTargetType.FlopCard3]
TABLE_CARDS = FLOP_CARDS + \
    [PokerTargetType.TurnCard, PokerTargetType.RiverCard]
ALL_CARDS = PLAYER_CARDS + TABLE_CARDS
