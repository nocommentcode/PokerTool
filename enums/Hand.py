from enums.Card import Card


class Hand:
    def __init__(self, card_1: Card, card_2: Card):
        self.card_1 = card_1
        self.card_2 = card_2

    def cards(self):
        return self.card_1, self.card_2

    def is_suited(self):
        return self.card_1.suit == self.card_2.suit

    def is_pokets(self):
        return self.card_1.value == self.card_2.value

    def __str__(self) -> str:
        return f"{str(self.card_1)} {str(self.card_2)}"
