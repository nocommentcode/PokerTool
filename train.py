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
DATASET_SUBSET = [TargetType.Player_card_1, TargetType.Player_card_2]
BATCH_SIZE = 32
EPOCHS = 350
SAVE_NAME = "6_player_pc_dp"
LR = 0.0007
CONV_CHANNELS = [16, 32]
FC_LAYER = [64]


def run(model, dataloader, device, is_train):
    with tqdm(dataloader, unit="batch") as tepoch:
        log = PokerNetworkLog({})
        for x, y in tepoch:
            tepoch.set_description("train" if is_train else "test")
            x = x.to(device)
            y.to(device)

            if is_train:
                batch_log = model.train_iter(x, y)
            else:
                batch_log = model.test_iter(x, y)

            log += batch_log
            tepoch.set_postfix(**batch_log)

        log /= len(dataloader)
        return log


def get_loaders(writer):
    dataset_dir = Path(os.path.join("images", "datasets", DATASET_NAME))
    train_loader, test_loader = data_loader_factory(
        dataset_dir, 0.7, batch_size=BATCH_SIZE, subsets=DATASET_SUBSET)

    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    print(f"{num_train} train samples, {num_test} test samples")

    log_distributions(train_loader, test_loader, writer)

    return train_loader, test_loader


def set_seeds():
    torch.manual_seed(1)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)


if __name__ == "__main__":
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter()

    train_loader, test_loader = get_loaders(writer)

    network = PokerNetwork(
        lr=LR, conv_channels=CONV_CHANNELS, fc_layers=FC_LAYER)
    network.to(device)

    print(f"Training on {device} for {EPOCHS}")
    for e in range(EPOCHS):
        print(f"\nEpoch {e+1}")
        network.train()
        train_log = run(network, train_loader, device, is_train=True)
        train_log.log(writer, e)

        network.eval()
        with torch.no_grad():
            test_log = run(network, test_loader,  device, is_train=False)
            test_log.log(writer, e)

    network.save(SAVE_NAME)
