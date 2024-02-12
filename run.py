import torch
from networks.model_factory import model_factory
from poker.StateProvider import StateProvider
from enums.GameType import GameType
from time import sleep
import msvcrt

GAME_TYPE = GameType.EightPlayer
AVAILABLE_BLINDS = [10, 12, 15, 20, 30, 40, 60, 80]
SAVE_SCREENSHOTS = False


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
    state_provider = StateProvider(state_detector, model, GAME_TYPE)

    while True:
        state_provider.tick(save_screenshots=SAVE_SCREENSHOTS)

        sleep(1)

        try:
            new_blinds = check_change_blinds()
            if new_blinds in AVAILABLE_BLINDS:
                print(f"Changing blinds to {new_blinds}")
                state_provider.set_blinds(new_blinds)

        except Exception as e:
            print(f"Error - {str(e)}")


if __name__ == "__main__":
    with torch.no_grad():
        run()
