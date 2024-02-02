import numpy as np
import torch
from torch.utils.data import DataLoader
from data.PokerTargetBatch import PokerTargetBatch
from enums.Suit import Suit
from enums.PokerTargetType import PokerTargetType
from enums.Value import Value
import matplotlib.pyplot as plt


class PokerCardDistribution:
    def __init__(self, type: PokerTargetType, name: str):
        self.type = type
        self.name = name
        self.suit_counts = [0 for _ in Suit]
        self.value_counts = [0 for _ in Value]
        self.total_samples = 0

    def __add__(self, other):

        if type(other) == PokerTargetBatch:
            suits, values, _ = other[self.type]
            for suit in suits:
                self.suit_counts[suit] += 1
            for value in values:
                self.value_counts[value] += 1
            self.total_samples += len(other)

        if type(other) == PokerTargetType:
            suit, value, _ = other[self.type]
            self.suit_counts[suit] += 1
            self.value_counts[value] += 1
            self.total_samples += 1

        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def __str__(self):
        string = self.type.value + ": \n"

        for suit, count in zip(Suit, self.suit_counts):
            percent = round(count * 100 / self.total_samples, 2)
            string += f"{suit}: {percent}%\n"

        string += "\n"

        for value, count in zip(Value, self.value_counts):
            percent = round(count * 100 / self.total_samples, 2)
            string += f"{value}: {percent}%\n"

        return string

    def log(self, writer):
        writer.add_text(
            f"{self.type.value}/Distribution/{self.name}", str(self))

    def get_class_weights(self):
        suits_counts = np.array(self.suit_counts)
        value_counts = np.array(self.value_counts)
        return torch.tensor(1 - (suits_counts/self.total_samples)).float().to('cuda'), torch.tensor(1-(value_counts/self.total_samples)).float().to('cuda')


def plot_dist(train, test, labels, ax):
    ind = np.arange(len(labels))
    width = 0.3

    ax.bar(ind, train, width, label="train")
    ax.bar(ind + width, test, width, label="test")
    ax.set_xticks(ind + width / 2, labels)
    ax.legend()


def plot_card_dist(train_dist, test_dist):
    fig, (suit_ax, val_ax) = plt.subplots(2, 1)

    suits = [str(suit) for suit in Suit]
    suit_train_percents = [round(count * 100 / train_dist.total_samples, 2)
                           for count in train_dist.suit_counts]
    suit_test_percents = [round(count * 100 / test_dist.total_samples, 2)
                          for count in test_dist.suit_counts]
    plot_dist(suit_train_percents, suit_test_percents, suits, suit_ax)

    values = [str(value) for value in Value]
    value_train_percents = [round(count * 100 / train_dist.total_samples, 2)
                            for count in train_dist.value_counts]
    value_test_percents = [round(count * 100 / test_dist.total_samples, 2)
                           for count in test_dist.value_counts]
    plot_dist(value_train_percents, value_test_percents, values, val_ax)

    return fig


def get_distributions_for_loader(loader: DataLoader):
    distributions = {type: PokerCardDistribution(
        type, "train") for type in PokerTargetType}
    for _, y in loader:
        for dist in distributions.values():
            dist += y
    return distributions


def get_distributions(train_loader: DataLoader, test_loader: DataLoader):
    return get_distributions_for_loader(train_loader), get_distributions_for_loader(test_loader)


def log_distributions(train_distributions, test_distributions, writer, name="distribution"):
    types = train_distributions.keys()
    for type in types:
        train = train_distributions[type]
        test = test_distributions[type]
        plot = plot_card_dist(train, test)
        writer.add_figure(f"{type.value}/{name}", plot)
