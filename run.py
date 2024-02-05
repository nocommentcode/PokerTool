import torch
from networks.model_factory import model_factory
from poker.StateProvider import StateProvider
from enums.GameType import GameType
from networks.StateDetector import StateDetector
from time import sleep

from networks.PokerNetwork import PokerNetwork
from ranges.RangeChart import load_range_charts


GAME_TYPE = GameType.SixPlayer


def run():
    state_detector, model = model_factory(GAME_TYPE)

    charts = load_range_charts()
    charts = charts[GAME_TYPE]

    state_provider = StateProvider(state_detector, model, GAME_TYPE, charts)

    while True:
        state_provider.tick(save_screenshots=True)
        sleep(1)


if __name__ == "__main__":
    with torch.no_grad():
        run()
