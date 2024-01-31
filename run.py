from PIL.Image import open as open_image
import os
import torch
from enums.GameType import GameType
from networks.CardDetector import CardDetector
from time import sleep
import pyautogui
import uuid
from data import CLASSIFIED_DIR, UN_CLASSIFIED_DIR
import numpy as np
import pandas as pd

from networks.PokerNetwork import PokerNetwork
from poker.GameStateFactory import GameStateFactory
from ranges.RangeChart import load_range_charts

MODEL_NAME = '6_player_pc_dp'
CARD_DETECTOR_NAME = "card_detector"
GAME_TYPE = GameType.SixPlayer


def take_screenshot():
    # return pyautogui.screenshot()
    return open_image(os.path.join(UN_CLASSIFIED_DIR, '3e1597ea-8e32-4cef-9453-dd276c5039f5.png'))


def run():
    card_detector = CardDetector.load(CARD_DETECTOR_NAME)
    card_detector.eval()

    model = PokerNetwork.load(MODEL_NAME, conv_channels=[
                              16, 32], fc_layers=[64])
    model.eval()

    charts = load_range_charts()
    charts = charts[GAME_TYPE]
    gs_fact = GameStateFactory(model, card_detector, GAME_TYPE, charts)

    current_player_count, current_table_count = 1, 0

    while True:
        screenshot = take_screenshot()
        pred_player_count, pred_table_count = card_detector.get_card_counts(
            screenshot)

        if pred_player_count != current_player_count or pred_table_count != current_table_count:
            print(str(gs_fact.get_game_state(screenshot,
                  pred_player_count, pred_table_count)))
            current_player_count, current_table_count = pred_player_count, pred_table_count

        sleep(1)


if __name__ == "__main__":
    with torch.no_grad():
        run()
