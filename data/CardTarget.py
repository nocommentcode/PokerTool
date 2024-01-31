from enums.Suit import Suit
from enums.Value import Value


class CardTarget:
    def __init__(self, label: str, uuid: str):
        if label == '':
            self.suit = Suit.Empty
            self.value = Value.Empty
            self.uuid = uuid
            return

        self.value = Value.from_string(label[0])
        self.suit = Suit.from_string(label[1])
        self.uuid = uuid

    def get_value_index(self):
        return self.value.value

    def get_suit_index(self):
        return self.suit.value
