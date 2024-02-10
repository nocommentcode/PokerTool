from enum import Enum


class Suit(Enum):
    Empty = 0
    Spades = 1
    Hearts = 2
    Diamonds = 3
    Clubs = 4

    @staticmethod
    def from_string(string: str) -> "Suit":
        if string == 'S':
            return Suit.Spades

        if string == 'D':
            return Suit.Diamonds

        if string == 'H':
            return Suit.Hearts

        if string == 'C':
            return Suit.Clubs

        raise AttributeError(f"Suit {string} does not exist")

    @staticmethod
    def from_index(index: int):
        suits = [Suit.Empty, Suit.Spades,
                 Suit.Hearts, Suit.Diamonds, Suit.Clubs]
        return suits[index]

    def __str__(self):
        symbols = ['', '♠', '♥', '♦', '♣']
        return symbols[self.value]

    def __eq__(self, other):
        if type(other) != Suit:
            return False

        return self.value == other.value

    def to_non_symbol_string(self):
        if self.value == 0:
            return ''

        if self.value == 1:
            return "S"

        if self.value == 2:
            return "H"

        if self.value == 3:
            return "D"

        if self.value == 4:
            return "C"

    def __gt__(self, other):
        if type(other) != Suit:
            return False
        # spade > club
        # heart > club
        # diamond > club
        # diamond > heart
        # spade > heart
        # spade > club
        # spade > diamond
        suit_order = [Suit.Clubs.value,
                      Suit.Hearts.value,
                      Suit.Diamonds.value,
                      Suit.Spades.value]
        my_index = suit_order.index(self.value)
        other_index = suit_order.index(other.value)
        return my_index > other_index


SUITS = [Suit.Empty, Suit.Spades, Suit.Hearts, Suit.Diamonds, Suit.Clubs]
