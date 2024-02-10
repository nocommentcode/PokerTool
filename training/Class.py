import torch
from tqdm import tqdm
from data.PokerCardDistribution import log_distributions
from enums.PokerTargetType import PokerTargetType
from networks.PokerNetwork import PokerNetwork


from typing import List
from networks.PokerNetworkLog import PokerNetworkLog

from training.Trainer import Trainer


class Class(Trainer):
    def __init__(self, name: str, subsets: List[PokerTargetType], train_encoder: bool, dataset_name: str, batch_size: int, epochs: int, lr: float):
        super().__init__(dataset_name, batch_size, epochs, lr, subsets)
        self.name = name
        self.train_encoder = train_encoder
        self.epoch_offset = 0

    def set_epoch_offset(self, offset):
        self.epoch_offset = offset

    def submit_log(self, log, writer, epoch):
        log.log(writer, epoch + self.epoch_offset)

    def freeze_model_layers(self, model: PokerNetwork):
        model.freeze_all()
        model.unfreeze(self.subsets)
        if self.train_encoder:
            model.unfreeze_encoder()

        non_frozen_parameters = [
            p for p in model.parameters() if p.requires_grad]
        model.optim = torch.optim.RMSprop(non_frozen_parameters, lr=self.lr)

    def log_loader_distributions(self, train_loader, test_loader, writer):
        log_distributions(train_loader, test_loader, writer,
                          f"{self.name} distribution")
