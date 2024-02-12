import torch
from enums.GameType import GameType
from torch.utils.tensorboard import SummaryWriter
from networks.model_factory import model_factory
from training.Tester import Tester


GAME_TYPE = GameType.EightPlayer
DATASET_NAME = "8_player"


if __name__ == "__main__":
    _, model = model_factory(GAME_TYPE)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device} for {DATASET_NAME}")

    writer = SummaryWriter()
    model.to(device)

    tester = Tester(DATASET_NAME, 32, 1,
                    0.1, None, weigh_loss=False)
    tester.start(model, writer, device)
