from enums.Suit import Suit
from enums.Value import Value
from utils.printing import black_text, red_text


class Card:
    def __init__(self, suit: Suit, value: Value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return self.coloured(f" {self.value}{self.suit} ")

    def coloured(self, text):
        if self.suit == Suit.Spades or self.suit == Suit.Clubs:
            func = black_text
        else:
            func = red_text

        return func(text, highlight_color="on_white")

    def __eq__(self, other):
        if type(other) != Card:
            return False

        return self.suit == other.suit and self.value == other.value

    def __gt__(self, other):
        if type(other) != Card:
            return False

        if self.value == other.value:
            return self.suit > other.suit

        return self.value > other.value
