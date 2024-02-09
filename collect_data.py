import torch
from enums.GameType import GameType
from networks.StateDetector import StateDetector
from time import sleep
import pyautogui
import uuid
from data import UN_CLASSIFIED_DIR
import numpy as np
import pandas as pd

from networks.PokerNetwork import PokerNetwork

GAME_TYPE = GameType.SixPlayer


def save_screenshot(screenshot):
    filename = f"{UN_CLASSIFIED_DIR}/{uuid.uuid4()}.png"
    screenshot.save(filename)


def take_screenshot():
    return pyautogui.screenshot()


def run():
    card_detector = StateDetector.load(STATE_DECTOR_NAME)
    card_detector.eval()

    total_counts = np.zeros((3, 6))

    def record_and_display_count(player, table):
        total_counts[player, table] += 1
        df = pd.DataFrame(total_counts, [0, 1, 2], [0, 1, 2, 3, 4, 5])
        print(player, table)
        print(df.head(3))
        print(f"Total: {total_counts.sum()} samples")
        print("\n\n\n")

    current_player_count, current_table_count = 0, 0

    while True:
        screenshot = take_screenshot()
        state = card_detector.get_state(
            screenshot)

        if state.player_card_count != current_player_count or state.table_card_count != current_table_count:
            save_screenshot(screenshot)
            record_and_display_count(
                state.player_card_count, state.table_card_count)
            current_player_count, current_table_count = state.player_card_count, state.table_card_count

        sleep(1)


if __name__ == "__main__":
    with torch.no_grad():
        run()
