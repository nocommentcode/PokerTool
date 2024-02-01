from data.CardTarget import CardTarget
from enums.PokerTargetType import PokerTargetType


class PokerTarget:
    def __init__(self, classification: str, uuid: str):
        cards = classification.split(',')

        targets = {}
        self.uuid = uuid

        for target_type, card_label in zip(PokerTargetType, cards):
            targets[target_type] = CardTarget(card_label, uuid)

        self.targets = targets

    def __getitem__(self, target_type):
        target = self.targets[target_type]
        return target.get_suit_index(), target.get_value_index(), self.uuid
