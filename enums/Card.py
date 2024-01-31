from enums.Suit import Suit
from enums.Value import Value


class Card:
    def __init__(self, suit: Suit, value: Value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value}{self.suit}"

    def __eq__(self, other):
        if type(other) != Card:
            return False

        return self.suit == other.suit and self.value == other.value
