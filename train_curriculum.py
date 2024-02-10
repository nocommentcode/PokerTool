import torch
from tqdm import tqdm
from training.Class import Class
from training.Curriculum import Curriculum
from enums.PokerTargetType import ALL_CARDS, FLOP_CARDS, PLAYER_CARDS, PokerTargetType
from networks.PokerNetwork import PokerNetwork
from torch.utils.tensorboard import SummaryWriter
from networks.PokerNetworkLog import PokerNetworkLog

DATASET_NAME = "6_player"
BATCH_SIZE = 32
EPOCHS = 350
SAVE_NAME = "6_player_player_cards"
LR = 0.001
CONV_CHANNELS = [32, 64]
FC_LAYER = [64]

base_params = {
    "dataset_name": DATASET_NAME,
    "batch_size": BATCH_SIZE,
    "lr": LR,
}
CLASSES = (
    Class(name="Player Cards", epochs=150, subsets=PLAYER_CARDS,
          train_encoder=True, **base_params),
    Class(name="Flop Cards", epochs=100,
          subsets=FLOP_CARDS, train_encoder=True, **base_params),
    Class(name="Turn Card", epochs=75,  subsets=[
          PokerTargetType.TurnCard], train_encoder=True, **base_params),
    Class(name="River Card", epochs=75,  subsets=[
          PokerTargetType.RiverCard], train_encoder=True, **base_params),
    Class(name="Combined", epochs=100,  subsets=ALL_CARDS,
          train_encoder=True, **base_params)
)


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

    model = PokerNetwork(
        lr=LR, conv_channels=CONV_CHANNELS, fc_layers=FC_LAYER)
    model.to(device)

    curriculum = Curriculum(CLASSES)
    curriculum.run(model, writer, device)

    model.save(SAVE_NAME)
