from os import listdir
from pathlib import Path
import random
from matplotlib import pyplot as plt
from data.PokerTarget import PokerTarget
from enums.TargetType import TargetType
from networks.PokerNetwork import PokerNetwork
from data.img_transformers import poker_img_transformer
from PIL.Image import open as image_open

from poker.GameStateFactory import GameStateFactory

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

    dir = Path("images/classified_images")
    remaining = list(listdir(dir))
    for fnamne in remaining:

        # get game state
        with open(Path(f"{dir}/{fnamne}/classification.txt")) as f:
            classi = f.read()
            target = PokerTarget(classi, '')
            suit1, value1, _ = target[TargetType.Player_card_1]
            _, value2, _ = target[TargetType.Player_card_2]
            problem_values = [1, 4, 9, 13]
            # if value1 not in problem_values or value2 not in problem_values:
            if suit1 != 3:
                continue
            print(fnamne)
            print(classi)

        # show
        image = image_open(Path(f"{dir}/{fnamne}/image.png"))
        plt.imshow(image)
        plt.show()
