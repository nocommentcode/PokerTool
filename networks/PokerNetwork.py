import torchvision
import os
from typing import Dict
from data.img_transformers import FINAL_DIMENSIONS
from networks import BASE_WIEGHT_DIR
from networks.CardNetworkHead import CardNetworkHead
from networks.IntNetworkHead import IntNetworkHead
from data.PokerTargetBatch import PokerTargetBatch
from enums.TargetType import CARD_TARGETS, INT_TARGETS, TargetType

import torch
import torch.nn as nn

from networks.NetworkHeadStep import NetworkHeadStep


class PokerNetworkLog(dict):
    def __init__(self, dict):
        for key, value in dict.items():
            self[key] = value

    def __add__(self, other):
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other

        for type in self.keys():
            other_result = other[type]
            self[type] += other_result

        return self

    def __truediv__(self, value):
        for type in self.keys():
            self[type] /= value

        return self

    def log(self, writer, epoch, images=True):
        for type, log in self.items():
            log_items = log.get_logs()
            for name, value in log_items:
                writer.add_scalar(f"{type}/{name}", value, epoch)

            if images and epoch % 50 == 0:
                log_images = log.get_images()
                for name, img in log_images:
                    writer.add_figure(f"{type}/{name}", img, epoch)


class PokerNetwork(nn.Module):
    def __init__(self, input_shape=FINAL_DIMENSIONS, lr=0.001, conv_channels=[32, 64, 128], fc_layers=[256]):
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

        # self.encoder = torchvision.models.resnet152(
        #     weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
        # self.encoder.fc = nn.Identity()
        # for p in self.encoder.parameters():
        #     p.requires_grad = False

        input = torch.zeros((1, *input_shape))
        output = self.encoder(input)

        output_dim = output.shape[1]
        nets = {}

        nets[TargetType.Dealer_pos.value] = IntNetworkHead(
            TargetType.Dealer_pos, output_dim, 9)
        # nets[TargetType.Num_players.value] = IntNetworkHead(
        #     TargetType.Num_players, output_dim, 10)

        # , TargetType.Flop_card_1, TargetType.Flop_card_2, TargetType.Flop_card_3]:
        for type in [TargetType.Player_card_1, TargetType.Player_card_2]:
            nets[type.value] = CardNetworkHead(type, output_dim, fc_layers)

        self.nets = torch.nn.ModuleDict(nets)

        # self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.optim = torch.optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)

        predictions = {type: net(encoded)
                       for type, net in self.nets.items()}

        return predictions

    def train_iter(self, x: torch.Tensor, target: PokerTargetBatch) -> PokerNetworkLog:
        encoded = self.encoder(x)

        loss_logs = {type: net.train_iter(encoded, target)
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
