import os
from typing import Dict, List
from data.img_transformers import CARDS_FINAL_DIMENTIONS
from enums.PokerTargetType import PLAYER_CARDS, ALL_CARDS, PokerTargetType
from networks import BASE_WIEGHT_DIR
from networks.CardNetworkHead import CardNetworkHead
from data.PokerTargetBatch import PokerTargetBatch

import torch
import torch.nn as nn

from networks.NetworkHeadStep import NetworkHeadStep
from networks.PokerNetworkLog import PokerNetworkLog


class PokerNetwork(nn.Module):
    def __init__(self, input_shape=CARDS_FINAL_DIMENTIONS, lr=0.001, conv_channels=[32, 64, 128], fc_layers=[256]):
        super().__init__()

        self.encoder = nn.Sequential()
        in_channels = 3
        for out_channel in conv_channels:
            self.encoder.append(nn.Conv2d(in_channels,  out_channel, 3))
            self.encoder.append(nn.Dropout())
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool2d(2))
            in_channels = out_channel

        self.encoder.append(nn.Flatten())

        input = torch.zeros((1, *input_shape))
        output = self.encoder(input)
        output_dim = output.shape[1]

        nets = {}
        for type in ALL_CARDS:
            nets[type.value] = CardNetworkHead(type, output_dim, fc_layers)

        self.nets = torch.nn.ModuleDict(nets)

        self.optim = torch.optim.RMSprop(
            self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)

        predictions = {type: net(encoded)
                       for type, net in self.nets.items()}

        return predictions

    def train_iter(self, x: torch.Tensor, target: PokerTargetBatch, loss_weights) -> PokerNetworkLog:
        encoded = self.encoder(x)

        loss_logs = {type: net.train_iter(encoded, target, loss_weights[type])
                     for type, net in self.nets.items()}
        total_loss = sum([loss for loss, _ in loss_logs.values()])

        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        return PokerNetworkLog(
            {type: log for type, (_, log) in loss_logs.items()})

    @torch.no_grad
    def test_iter(self, x: torch.Tensor, target: PokerTargetBatch) -> NetworkHeadStep:
        encoded = self.encoder(x)

        test_results = {type: net.test_iter(encoded, target)
                        for type, net in self.nets.items()}

        return PokerNetworkLog(test_results)

    def log(self, step: NetworkHeadStep):
        for name, value in step.get_logs():
            self.writer.add_scalar(name, value, self.global_step)
            self.global_step += 1

    def save(self, filename: str):
        filename = os.path.join(BASE_WIEGHT_DIR, f'{filename}.pth')
        torch.save(self.state_dict(), filename)

    @torch.no_grad
    def predict(self, x: torch.Tensor):
        encoded = self.encoder(x)

        predictions = {type: net.predict(encoded)
                       for type, net in self.nets.items()}

        return predictions

    @staticmethod
    def load(filename: str, conv_channels=[32, 64, 128], fc_layers=[256]):
        filename = os.path.join(BASE_WIEGHT_DIR, f'{filename}.pth')
        model = PokerNetwork(conv_channels=conv_channels, fc_layers=fc_layers)
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict, assign=True)
        return model

    # def un_freeze_all(self):
    #     for parameter in self.parameters():
    #         parameter.requires_grad = True

    def freeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def unfreeze(self, targets: List[PokerTargetType]):
        for target in targets:
            for parameter in self.nets[target.value].parameters():
                parameter.requires_grad = True
