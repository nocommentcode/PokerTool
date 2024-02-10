import torch
from networks.model_factory import model_factory
from poker.StateProvider import StateProvider
from enums.GameType import GameType
from time import sleep

from ranges.RangeChart import load_range_charts
import msvcrt

GAME_TYPE = GameType.NinePlayer
BLINDS = 20


def check_change_blinds():
    input = ""
    while msvcrt.kbhit():
        input += msvcrt.getwch()

    if len(input) < 3:
        return None

    if input[0] != "b":
        return None

    return int(input[1:])


def run():
    state_detector, model = model_factory(GAME_TYPE)

    charts = load_range_charts(GAME_TYPE, BLINDS)
    charts = charts

    state_provider = StateProvider(state_detector, model, GAME_TYPE, charts)

    while True:
        state_provider.tick(save_screenshots=True)

        sleep(1)

        try:
            new_blinds = check_change_blinds()
            if new_blinds is not None:
                print(f"Changing blinds to {new_blinds}")
                new_charts = load_range_charts(GAME_TYPE, new_blinds)
                state_provider.set_charts(new_charts)

        except Exception as e:
            print(f"Error - {str(e)}")


if __name__ == "__main__":
    with torch.no_grad():
        run()
