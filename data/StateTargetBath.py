import torch


class StateTargetBath:
    def __init__(self, targets):
        self.targets = targets
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, target_type):
        values = [t.targets[target_type] for t in self.targets]
        uuids = [t.uuid for t in self.targets]

        return torch.LongTensor(values).to(device=self.device), uuids
