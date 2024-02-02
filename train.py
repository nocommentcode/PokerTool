import torch
from enums.PokerTargetType import PokerTargetType
from networks.PokerNetwork import PokerNetwork
from torch.utils.tensorboard import SummaryWriter

from training.Trainer import Trainer

DATASET_NAME = "6_player"
DATASET_SUBSET = None
WEIGH_LOSSES = False
BATCH_SIZE = 32
EPOCHS = 350
SAVE_NAME = "6_player_player_cards"
LR = 0.0005
CONV_CHANNELS = [32, 64]
FC_LAYER = [64]


def set_seeds():
    torch.manual_seed(1)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)


if __name__ == "__main__":
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device} for {EPOCHS}")

    writer = SummaryWriter()

    model = PokerNetwork(
        lr=LR, conv_channels=CONV_CHANNELS, fc_layers=FC_LAYER)
    model.to(device)

    trainer = Trainer(DATASET_NAME, BATCH_SIZE, EPOCHS,
                      LR, DATASET_SUBSET, weigh_loss=WEIGH_LOSSES)
    trainer.start(model, writer, device)

    model.save(SAVE_NAME)
