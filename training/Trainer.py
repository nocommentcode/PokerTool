import os
from pathlib import Path
from typing import List
from data import DATASET_DIR
from data.PokerCardDistribution import log_distributions, get_distributions
from data.PokerDataset import data_loader_factory
from enums.PokerTargetType import PokerTargetType
from enums.Suit import Suit
from enums.Value import Value
from networks.PokerNetwork import PokerNetwork
from networks.PokerNetworkLog import PokerNetworkLog
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, dataset_name: str, batch_size: int, epochs: int, lr: float, class_equalizing_subsets: List[PokerTargetType] = None, weigh_loss=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dataset_name = dataset_name
        self.subsets = class_equalizing_subsets
        self.weigh_loss = weigh_loss

    def log_loader_distributions(self, train_dist, test_dist, writer):
        log_distributions(train_dist, test_dist, writer)

    def get_data_loaders(self, writer):
        dataset_dir = Path(os.path.join(DATASET_DIR, self.dataset_name))
        train_loader, test_loader = data_loader_factory(
            dataset_dir, 0.7, batch_size=self.batch_size, subsets=self.subsets)

        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        print(f"{num_train} train samples, {num_test} test samples")

        train_dist, test_dist = get_distributions(train_loader, test_loader)
        self.log_loader_distributions(train_dist, test_dist, writer)
        if self.weigh_loss:
            self.loss_weights = {type.value: dist.get_class_weights()
                                 for type, dist in train_dist.items()}
        else:
            self.loss_weights = {type.value: (
                None, None) for type in PokerTargetType}

        return train_loader, test_loader

    def submit_log(self, log, writer, epoch):
        log.log(writer, epoch)

    def start(self, model: PokerNetwork, writer, device):
        train_loader, test_loader = self.get_data_loaders(writer)
        print(self.loss_weights)

        for e in range(self.epochs):
            print(f"\nEpoch {e+1}")
            model.train()
            train_log = self.run_epoch(
                model, train_loader, device, is_train=True)
            self.submit_log(train_log, writer, e)

            model.eval()
            with torch.no_grad():
                test_log = self.run_epoch(model, test_loader,
                                          device, is_train=False)
                self.submit_log(test_log, writer, e)

    def run_epoch(self, model, dataloader, device, is_train):
        with tqdm(dataloader, unit="batch") as tepoch:
            log = PokerNetworkLog({})
            for x, y in tepoch:
                tepoch.set_description("train" if is_train else "test")
                x = x.to(device)
                y.to(device)

                if is_train:
                    batch_log = model.train_iter(x, y,  self.loss_weights)
                else:
                    batch_log = model.test_iter(x, y)

                log += batch_log
                tepoch.set_postfix(**batch_log)

            log /= len(dataloader)
            return log
