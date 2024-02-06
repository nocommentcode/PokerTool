from os import listdir
from pathlib import Path
import random
from matplotlib import pyplot as plt
from PIL.Image import open as image_open
from enums.PokerTargetType import PokerTargetType

from enums.GameType import GameType
from networks.model_factory import model_factory
from poker.StateProvider import StateProvider
from ranges.RangeChart import load_range_charts

GAME_TYPE = GameType.SixPlayer


if __name__ == "__main__":
    # model = PokerNetwork.load("6_player")
    # model.eval()
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # load an image
    # dir = Path("images/classified_images")
    # remaining = list(listdir(dir))
    # file = random.choice(remaining)
    # image = open(Path(f"{dir}/{file}/image.png"))

    # # get game state
    # gs_fact = GameStateFactory(model, 'cuda')
    # gs = gs_fact.proccess_screenshot(image)
    # print(gs)

    # # show
    # from torchvision import transforms
    # poker_img_transformer.transforms.append(transforms.ToPILImage())
    # transformed = poker_img_transformer(image)
    # plt.imshow(image)
    # plt.show()
    state_detector, model = model_factory(GAME_TYPE)

    charts = load_range_charts()
    charts = charts[GAME_TYPE]

    state_provider = StateProvider(state_detector, model, GAME_TYPE, charts)
    dir = Path("images/classified_images")
    remaining = list(listdir(dir))[23:]
    for fnamne in remaining:

        # get game state
        # with open(Path(f"{dir}/{fnamne}/classification.txt")) as f:
        #     classi = f.read()
        #     print(classi)

        # show
        image = image_open(Path(f"{dir}/{fnamne}/image.png"))
        state_provider.print_for_screenshot_(image)

        plt.imshow(image)
        plt.show()
