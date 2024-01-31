import numpy as np
import os
from pathlib import Path
import torch

from tqdm import tqdm
from data.PokerCardDistribution import log_distributions

from data.PokerDataset import data_loader_factory
from enums.TargetType import TargetType
from networks.PokerNetwork import PokerNetwork, PokerNetworkLog
from torch.utils.tensorboard import SummaryWriter

DATASET_NAME = "6_player"

BATCH_SIZE = [6, 12, 24, 32, 64]

EPOCHS = 200

LR = [0.01, 0.001, 0.0001]

CONVOLUTIONS = [
    (8, ),
    (8, 16),
    (8, 16, 32),
    (32, 64),
    (64, ),
    (32, 64, 128),
    (128, )
]

FC_LAYERS = [
    (56,),
    (56, 128),
    (128, ),
    (128, 256),
]


def set_seeds():
    torch.manual_seed(1)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)


def run(model, dataloader, device, is_train):
    log = PokerNetworkLog({})
    for x, y in dataloader:
        x = x.to(device)
        y.to(device)

        if is_train:
            batch_log = model.train_iter(x, y)
        else:
            batch_log = model.test_iter(x, y)

        log += batch_log

    log /= len(dataloader)
    return log


def get_loaders(writer, batch_size):
    dataset_dir = Path(os.path.join("images", "datasets", DATASET_NAME))
    train_loader, test_loader = data_loader_factory(
        dataset_dir, 0.7, batch_size=batch_size)

    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    print(f"{num_train} train samples, {num_test} test samples")

    return train_loader, test_loader


def train(lr, batch_size, fc, conv):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter()

    train_loader, test_loader = get_loaders(writer, batch_size)

    network = PokerNetwork(lr=lr, conv_channels=conv, fc_layers=fc)
    network.to(device)

    print(f"{lr}, {batch_size}, {fc}, {conv}")

    for e in range(EPOCHS):
        network.train()
        train_log = run(network, train_loader, device, is_train=True)
        train_log.log(writer, e, images=False)

        network.eval()
        with torch.no_grad():
            test_log = run(network, test_loader,  device, is_train=False)
            test_log.log(writer, e, images=False)

    def print_log(target_type):
        logs = test_log[target_type.value].get_logs()
        print(target_type.value)
        for name, value in logs:
            print(f"{name}: {value}")

    print_log(TargetType.Player_card_1)
    print_log(TargetType.Player_card_2)

    print("\n\n\n")


if __name__ == "__main__":
    num_searches = 200

    indexes = np.random.randint(low=[0 for _ in range(4)],
                                high=(len(LR), len(BATCH_SIZE), len(
                                    CONVOLUTIONS), len(FC_LAYERS)),
                                size=(num_searches, 4))

    attempts = []
    for index in indexes:
        lr, batch, conv, fc = index
        if (lr, batch, conv, fc) in attempts:
            continue
        attempts.append((lr, batch, conv, fc))
        try:
            train(LR[lr], BATCH_SIZE[batch], FC_LAYERS[fc], CONVOLUTIONS[conv])
        except Exception as e:
            print(e)
