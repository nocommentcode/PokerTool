from enum import Enum


class Value(Enum):
    Empty = 0
    Ace = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Jack = 11
    Queen = 12
    King = 13

    @staticmethod
    def from_string(string: str) -> "Value":
        values = {"": None,
                  "A": Value.Ace,
                  "2": Value.Two,
                  "3": Value.Three,
                  "4": Value.Four,
                  "5": Value.Five,
                  "6": Value.Six,
                  "7": Value.Seven,
                  "8": Value.Eight,
                  "9": Value.Nine,
                  "T": Value.Ten,
                  "J": Value.Jack,
                  "Q": Value.Queen,
                  "K": Value.King}

        return values[string]

    @staticmethod
    def from_index(index: int):
        values = [Value.Empty,
                  Value.Ace,
                  Value.Two,
                  Value.Three,
                  Value.Four,
                  Value.Five,
                  Value.Six,
                  Value.Seven,
                  Value.Eight,
                  Value.Nine,
                  Value.Ten,
                  Value.Jack,
                  Value.Queen,
                  Value.King]

        return values[index]

    def __str__(self):
        symbols = [
            "",
            "A",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "T",
            "J",
            "Q",
            "K"
        ]
        return symbols[self.value]

    def __eq__(self, other):
        if type(other) != Value:
            return False

        return self.value == other.value

    def __gt__(self, other):
        if type(other) != Value:
            return False

        # handle aces
        if self.value == 1 and other.value != 1:
            return True
        if self.value != 1 and other.value == 1:
            return False

        return self.value > other.value


VALUES = [Value.Empty,
          Value.Ace,
          Value.Two,
          Value.Three,
          Value.Four,
          Value.Five,
          Value.Six,
          Value.Seven,
          Value.Eight,
          Value.Nine,
          Value.Ten,
          Value.Jack,
          Value.Queen,
          Value.King]
