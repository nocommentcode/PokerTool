import torch
from networks.CardDetector import CardDetector
from time import sleep
import pyautogui
import uuid
from data import UN_CLASSIFIED_DIR
import numpy as np
import pandas as pd

from networks.PokerNetwork import PokerNetwork
from poker.GameStateFactory import GameStateFactory


def save_screenshot(screenshot):
    filename = f"{UN_CLASSIFIED_DIR}/{uuid.uuid4()}.png"
    screenshot.save(filename)


def take_screenshot():
    return pyautogui.screenshot()


def run():
    card_detector = CardDetector.load("card_detector")
    card_detector.eval()

    model = PokerNetwork.load("6_player", conv_channels=[
                              16, 32], fc_layers=[40])
    model.eval()

    gs_fact = GameStateFactory(model, 'cuda')

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
        pred_player_count, pred_table_count = card_detector.get_card_counts(
            screenshot)

        if pred_player_count != current_player_count or pred_table_count != current_table_count:
            save_screenshot(screenshot)
            record_and_display_count(pred_player_count, pred_table_count)
            print(str(gs_fact.proccess_screenshot(screenshot)))
            current_player_count, current_table_count = pred_player_count, pred_table_count

        sleep(1)


if __name__ == "__main__":
    with torch.no_grad():
        run()
