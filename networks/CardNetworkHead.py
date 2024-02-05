from data.PokerTargetBatch import PokerTargetBatch
from enums.Suit import Suit
from enums.PokerTargetType import PokerTargetType
from enums.Value import Value
import torch
import torch.nn as nn
from networks.CardHeadLog import CardHeadLog
from networks.CardNetworkPrediction import CardNetworkPrediction


class CardNetworkHead(nn.Module):
    def __init__(self, target_type: PokerTargetType, dims: int, fc_layers=[256]):
        super().__init__()

        self.suit_net = nn.Sequential()
        in_neurons = dims
        for neurons in fc_layers:
            self.suit_net.append(nn.Linear(in_neurons, neurons))
            self.suit_net.append(nn.ReLU())
            self.suit_net.append(nn.Dropout())
            in_neurons = neurons
        self.suit_net.append(nn.Linear(in_neurons, len(Suit)))

        self.value_net = nn.Sequential()
        in_neurons = dims
        for neurons in fc_layers:
            self.value_net.append(nn.Linear(in_neurons, neurons))
            self.value_net.append(nn.ReLU())
            self.value_net.append(nn.Dropout())
            in_neurons = neurons
        self.value_net.append(nn.Linear(in_neurons, len(Value)))

        self.target_type = target_type

    def forward(self, x: torch.Tensor):
        suit = self.suit_net(x)
        value = self.value_net(x)

        return suit, value

    def train_iter(self, x: torch.Tensor, target: PokerTargetBatch, loss_weights=(None, None)) -> CardHeadLog:
        suit_weights, value_weights = loss_weights

        pred_suit, pred_value = self.forward(x)
        act_suit, act_value, uuids = target[self.target_type]

        suit_loss_fc = nn.CrossEntropyLoss(suit_weights)
        suit_loss = suit_loss_fc(pred_suit, act_suit)

        value_loss_fc = nn.CrossEntropyLoss(value_weights)
        value_loss = value_loss_fc(pred_value, act_value)

        return (suit_loss+value_loss), CardHeadLog("train", suit_loss.detach().item(), value_loss.detach().item(), pred_suit, act_suit, pred_value, act_value, uuids)

    @torch.no_grad
    def test_iter(self, x: torch.Tensor, target: PokerTargetBatch) -> CardHeadLog:
        pred_suit, pred_value = self.forward(x)
        act_suit, act_value, uuids = target[self.target_type]

        loss_fn = nn.CrossEntropyLoss()
        suit_loss = loss_fn(pred_suit, act_suit)
        value_loss = loss_fn(pred_value, act_value)

        return CardHeadLog("test", suit_loss.item(), value_loss.item(), pred_suit, act_suit, pred_value, act_value, uuids)

    @torch.no_grad
    def predict(self, x: torch.Tensor) -> CardNetworkPrediction:
        pred_suit, pred_value = self.forward(x)

        def predict(pred):
            softmax = nn.Softmax(dim=1)
            classes = torch.argmax(softmax(pred), 1)
            return classes

        return CardNetworkPrediction(predict(pred_suit).long().detach().cpu().numpy(), predict(pred_value).long().detach().cpu().numpy())
