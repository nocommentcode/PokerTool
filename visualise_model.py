from os import listdir
from pathlib import Path
from matplotlib import pyplot as plt
from PIL.Image import open as image_open
from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.Position import Position
from networks.model_factory import model_factory
from poker.Player import Player
from poker.StateProvider import StateProvider

GAME_TYPE = GameType.EightPlayer
GAME_STAGES = [GameStage.PREFLOP, GameStage.FLOP,
               GameStage.Turn, GameStage.River]

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    state_detector, model = model_factory(GAME_TYPE)

    state_provider = StateProvider(state_detector, model, GAME_TYPE)
    state_provider.blinds = 800

    dir = Path("images/unclassified_images")
    remaining = list(listdir(dir))
    remaining.sort(key=lambda x: os.path.getmtime(
        os.path.join(dir, f"{x}")))
    plt_object = None
    for fnamne in remaining[400:]:
        image = image_open(Path(f"{dir}/{fnamne}"))
        relevant = state_provider.print_for_screenshot_(
            image, game_stages=GAME_STAGES)

        if not relevant:
            continue

        if plt_object is None:
            plt_object = plt.imshow(image)
            plt.pause(0.01)
        else:
            plt_object.set_data(image)
            plt.draw()

        input("")
