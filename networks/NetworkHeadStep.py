import torch


class NetworkHeadStep:
    def __init__(self, loss: torch.Tensor, *logs: (str, float)):
        self.loss = loss
        self.logs = list(logs)
        self.__dict__ = self.to_dict()

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, NetworkHeadStep):
            raise AttributeError(
                f"type {other} cannot be added to NetworkHeadStep")
        self.loss += other.loss
        self.logs += other.logs
        self.__dict__ = self.to_dict()

        return self

    def __radd__(self, other):
        return self.__add__(other)

    def get_loss(self) -> torch.Tensor:
        return self.loss

    def get_logs(self):
        return self.logs

    def to_dict(self):
        return {name: value for name, value in self.logs}
