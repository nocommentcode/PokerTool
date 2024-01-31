from enums.TargetType import CARD_TARGETS, INT_TARGETS


import torch


class PokerTargetBatch:
    def __init__(self, targets):
        self.targets = targets
        self.device = 'cpu'

    def to(self, device):
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, target_type):
        if target_type in INT_TARGETS:
            return torch.LongTensor([t.targets[target_type].get_value() for t in self.targets]).to(device=self.device)

        if target_type in CARD_TARGETS:
            suits = [t.targets[target_type].get_suit_index()
                     for t in self.targets]
            values = [t.targets[target_type].get_value_index()
                      for t in self.targets]
            uuids = [t.targets[target_type].uuid for t in self.targets]

            return torch.LongTensor(suits).to(device=self.device), torch.LongTensor(values).to(device=self.device), uuids

        raise Exception(f"target {target_type} not supported")
