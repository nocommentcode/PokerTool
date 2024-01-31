import numpy as np
from torch.utils.data import DataLoader
from data.PokerTargetBatch import PokerTargetBatch
from data.PokerTarget import PokerTarget
from enums.Suit import Suit
from enums.TargetType import CARD_TARGETS, TargetType
from enums.Value import Value


class PokerCardDistribution:
    def __init__(self, type: TargetType, name: str):
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

        if type(other) == PokerTarget:
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


class PokerIntDistribution:
    def __init__(self, type: TargetType, name: str):
        self.type = type
        self.name = name
        self.counts = [0 for _ in range(14)]
        self.total_samples = 0

    def __add__(self, other):

        if type(other) == PokerTargetBatch:
            values = other[self.type]
            for value in values:
                self.counts[value] += 1
            self.total_samples += len(other)

        if type(other) == PokerTarget:
            value = other[self.type]
            self.counts[value] += 1
            self.total_samples += 1

        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def __str__(self):
        string = self.type.value + ": \n"

        for i, count in enumerate(self.counts):
            percent = round(count * 100 / self.total_samples, 2)
            string += f"{i}: {percent}%\n"
        string += "\n"

        return string

    def log(self, writer):
        writer.add_text(
            f"{self.type.value}/Distribution/{self.name}", str(self))


def get_distribution_on_data_loader(loader: DataLoader, type: TargetType):
    dist = PokerCardDistribution(type)
    for _, y in loader:
        dist += y

    print(str(dist))


def plot_dist(train, test, labels, ax):
    ind = np.arange(len(labels))
    width = 0.3

    ax.bar(ind, train, width, label="train")
    ax.bar(ind + width, test, width, label="test")
    ax.set_xticks(ind + width / 2, labels)
    ax.legend()


def plot_card_dist(train_dist, test_dist):
    import matplotlib.pyplot as plt
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


def plot_int_dist(train_dist, test_dist):
    import matplotlib.pyplot as plt
    fig, (ax) = plt.subplots(1, 1)
    labels = [i for i in range(len(train_dist.counts))]
    train_counts = [round(count * 100 / train_dist.total_samples, 2)
                    for count in train_dist.counts]
    test_counts = [round(count * 100 / test_dist.total_samples, 2)
                   for count in test_dist.counts]
    plot_dist(train_counts, test_counts, labels, ax)

    return fig


def log_distributions(train_loader: DataLoader, test_loader: DataLoader, writer):
    train_distributions = [PokerCardDistribution(
        type, "train") if type in CARD_TARGETS else PokerIntDistribution(type, "train") for type in TargetType]
    for _, y in train_loader:
        for dist in train_distributions:
            dist += y

    test_distributions = [PokerCardDistribution(
        type, "test") if type in CARD_TARGETS else PokerIntDistribution(type, "test") for type in TargetType]
    for _, y in test_loader:
        for dist in test_distributions:
            dist += y

    for train, test in zip(train_distributions, test_distributions):
        if train.type in CARD_TARGETS:
            plot = plot_card_dist(train, test)
        else:
            plot = plot_int_dist(train, test)

        writer.add_figure(f"{train.type.value}/distribution", plot)


def log_distribution(loader: DataLoader, name: str, writer):
    distributions = [PokerCardDistribution(
        type, name) if type in CARD_TARGETS else PokerIntDistribution(type, name) for type in TargetType]
    for _, y in loader:
        for dist in distributions:
            dist += y

    for dist in distributions:
        dist.log(writer)
