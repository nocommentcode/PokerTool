from data.CardTarget import CardTarget
from data.IntTarget import IntTarget
from enums.TargetType import CARD_TARGETS, TargetType


class PokerTarget:
    def __init__(self, classification: str, uuid: str):
        num_players, dealer_pos, *cards = classification.split(
            ',')

        targets = {}
        self.uuid = uuid
        targets[TargetType.Num_players] = IntTarget(num_players)
        targets[TargetType.Dealer_pos] = IntTarget(dealer_pos)

        for target_type, card_label in zip(CARD_TARGETS, cards):
            targets[target_type] = CardTarget(card_label, uuid)

        self.targets = targets

    def __getitem__(self, target_type):
        target = self.targets[target_type]
        if type(target) == IntTarget:
            return target.target, self.uuid

        if type(target) == CardTarget:
            return target.get_suit_index(), target.get_value_index(), self.uuid
