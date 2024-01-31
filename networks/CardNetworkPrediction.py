from enums.Card import Card
from enums.Suit import Suit
from enums.Value import Value


class CardNetworkPrediction:
    def __init__(self, pred_suit, pred_value):
        pred_suit_idx = pred_suit[0]
        self.suit = Suit.from_index(pred_suit_idx)

        pred_value_idx = pred_value[0]
        self.value = Value.from_index(pred_value_idx)

        self.card = Card(self.suit, self.value)
