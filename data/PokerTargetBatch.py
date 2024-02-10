
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
        suits = [t.targets[target_type].get_suit_index()
                 for t in self.targets]
        values = [t.targets[target_type].get_value_index()
                  for t in self.targets]
        uuids = [t.targets[target_type].uuid for t in self.targets]

        return torch.LongTensor(suits).to(device=self.device), torch.LongTensor(values).to(device=self.device), uuids
