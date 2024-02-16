from os import listdir
from pathlib import Path
import re
import time
from typing import Tuple
from matplotlib import pyplot as plt

import numpy as np
from enums.GameType import GameType
from enums.Position import Position
from PIL.Image import Image
import pytesseract
from PIL.Image import open as image_open

from enums.StackSize import StackSize
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

STACK_TOP_LEFT = {
    GameType.EightPlayer: [
        (782, 1162),
        (280, 990),
        (50, 666),
        (326, 370),
        (798, 302),
        (1290, 370),
        (1543, 666),
        (1312, 990),
    ]
}
STACK_DIMS = (225, 50)

CALL_RANGE_TOP_LEFT = {
    GameType.EightPlayer: [
        (788, 1079),
        (281, 915),
        (50, 589),
        (324, 294),
        (797, 225),
        (1290, 294),
        (1542, 589),
        (1312, 913),
    ]
}
CALL_RANGE_DIMS = (38, 29)


class Player:
    def __init__(self, game_type: GameType, blinds: int, index: int, position: Position, screenshot: Image):
        self.position = position
        self.stack_size = self.get_stack_size(
            screenshot, game_type, blinds, index)
        self.range = self.get_calling_range(screenshot, game_type, index)

    def extract_int(self, screenshot: Image, top_left: Tuple[int, int], dims: Tuple[int, int]) -> int:
        x, y = top_left
        dx, dy = dims
        image = screenshot.crop((x, y, x+dx, y+dy))
        value = pytesseract.image_to_string(image, config="--psm 13")

        try:
            int_value = re.sub("[^0-9]", "", value)
            return int(int_value)
        except Exception as e:
            return 0

    def get_stack_size(self, screenshot: Image, game_type: GameType, blinds: int, index: int) -> StackSize:
        top_left = STACK_TOP_LEFT[game_type][index]
        value = self.extract_int(screenshot, top_left, STACK_DIMS)
        self.stack_val = value
        return StackSize.from_val(value, blinds)

    def get_calling_range(self, screenshot: Image, game_type: GameType, index: int) -> int:
        top_left = CALL_RANGE_TOP_LEFT[game_type][index]
        return self.extract_int(screenshot, top_left, CALL_RANGE_DIMS)

    def __str__(self):
        return f"{self.position.pretty_str()} ({self.stack_size.pretty_str()})"


if __name__ == "__main__":
    dir = Path("images/unclassified_images")
    image = image_open(Path(f"{dir}/989c1ebd-9889-4edd-b05b-44db1e2723ca.png"))
    plt.imshow(image)
    plt.show()

    start = time.time()
    for i in range(8):
        p = Player(GameType.EightPlayer, 100, i, Position.BB, image)
        # print(p)

    print(f"took: {str(time.time()-start)}s")
