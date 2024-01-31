from data.PokerTargetBatch import PokerTargetBatch
from enums.TargetType import TargetType
import torch
import torch.nn as nn
from networks.IntHeadLog import IntHeadLog
from networks.IntNetworkPrediction import IntNetworkPrediction

from networks.NetworkHeadStep import NetworkHeadStep


class IntNetworkHead(nn.Module):
    def __init__(self, target_type: TargetType, n_inputs: int, n_ouputs: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_inputs, 10), nn.ReLU(), nn.Linear(10, n_ouputs))
        self.target_type = target_type
        self.n_ouputs = n_ouputs

    def forward(self, x: torch.Tensor):
        return self.layer(x)

    def train_iter(self, x: torch.Tensor, target: PokerTargetBatch) -> (torch.Tensor, IntHeadLog):
        pred = self.forward(x)
        actual = target[self.target_type]

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, actual)

        return loss, IntHeadLog("train", self.n_ouputs, loss.item(), pred, actual)

    @torch.no_grad
    def test_iter(self, x: torch.Tensor, target: PokerTargetBatch) -> IntHeadLog:
        pred = self.forward(x)
        actual = target[self.target_type]

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, actual)

        return IntHeadLog("test", self.n_ouputs, loss.item(),  pred, actual)

    @torch.no_grad
    def predict(self, x: torch.Tensor) -> IntNetworkPrediction:
        pred = self.forward(x)
        softmax = nn.Softmax(dim=1)
        classes = torch.argmax(softmax(pred), 1)

        return IntNetworkPrediction(classes.detach().cpu().numpy())
