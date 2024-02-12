from data import DATASET_DIR
from data.PokerDataset import data_loader_factory
from enums.PokerTargetType import PokerTargetType
from networks.PokerNetwork import PokerNetwork
from training.Trainer import Trainer


import torch


import os
from pathlib import Path


class Tester(Trainer):
    def get_data_loaders(self, writer):
        dataset_dir = Path(os.path.join(DATASET_DIR, self.dataset_name))
        _, test_loader = data_loader_factory(
            dataset_dir, 0.1, batch_size=self.batch_size, subsets=self.subsets)

        num_test = len(test_loader.dataset)
        print(f"Tesing on {num_test} samples")

        self.loss_weights = {type.value: (
            None, None) for type in PokerTargetType}

        return _, test_loader

    def start(self, model: PokerNetwork, writer, device):
        _, test_loader = self.get_data_loaders(writer)

        for e in range(1):
            model.eval()
            with torch.no_grad():
                test_log = self.run_epoch(model, test_loader,
                                          device, is_train=False)
                self.submit_log(test_log, writer, e)
