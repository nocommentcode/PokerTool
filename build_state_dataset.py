from PIL.Image import Image
from PIL.Image import fromarray
import numpy as np
from data.img_transformers import cards_transformer
import torch
from data.img_transformers import table_transformer
import datetime
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from data import DATASET_DIR, UN_CLASSIFIED_DIR
import os
from PIL.Image import open as open_image
from data.GGPokerHandHistory import GGPokerHandHistory
from enums.GameType import GameType
from enums.PokerTargetType import PLAYER_CARDS, TABLE_CARDS
from enums.Suit import Suit
from networks.PokerNetwork import PokerNetwork
from networks.model_factory import model_factory
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATASET_NAME = "8_player_state"
UNCLASSIDIED_NAME = UN_CLASSIFIED_DIR
# UNCLASSIDIED_NAME = "images/unclassified_images_from_big_batch_2"
GAME_TYPE = GameType.EightPlayer
DEBUG = True

OPPONENT_POSITIONS = {
    GameType.EightPlayer: (
        (387, 837),
        (151, 512),
        (430, 219),
        (899, 147),
        (1394, 220),
        (1644, 512),
        (1421, 843),
    )
}


class StateClassification:
    def __init__(self, uuid, hands):
        self.uuid = uuid
        self.file_path = os.path.join(
            UNCLASSIDIED_NAME, f"{self.uuid}.png")
        self.hand = self.get_corresponding_hand(hands)

    def get_corresponding_hand(self, hands):
        time_created = datetime.datetime.fromtimestamp(
            os.path.getmtime(self.file_path))

        hand_idx = 0
        current_hand = hands[hand_idx]
        next_hand = hands[hand_idx+1]

        while not (current_hand.start_time <= time_created <= next_hand.start_time):
            hand_idx += 1
            current_hand = hands[hand_idx]
            next_hand = hands[hand_idx+1]

        if (next_hand.start_time - time_created).total_seconds() < 3 or (time_created - current_hand.start_time).total_seconds() < 3:
            raise Exception(
                f"Screenshot time ({time_created}) too close to end of hand ({next_hand.start_time }), skipping...")

        return current_hand

    def save_to_dataset(self, image, classification):
        dir = Path(f"{DATASET_DIR}/{DATASET_NAME}/{self.uuid}")
        dir.mkdir(parents=True, exist_ok=True)

        tensor_img = table_transformer(image)
        torch.save(tensor_img, f"{dir}/image.pt")

        with open(f"{dir}/classification.txt", 'w') as f:
            f.write(classification)
            f.close()

    def sort_player_cards_from_hand(self):
        card_1, card_2 = self.hand.player_cards
        value_1, value_2 = card_1.value, card_2.value

        if value_1 == value_2:
            print(f"------- Pocket {str(value_1)} ({self.uuid}) ----------")
            suit_1, suit_2 = card_1.suit, card_2.suit
            if suit_1 > suit_2:
                return card_1, card_2

            return card_2, card_1

        if value_1 > value_2:
            return card_1, card_2

        return card_2, card_1

    def player_card_detection(self, image: Image):
        points = OPPONENT_POSITIONS[GAME_TYPE]

        upper = np.array([150, 80, 80], np.uint8)
        lower = np.array([112, 40, 40], np.uint8)
        pixels = np.asarray(image).copy()

        def player_has_cards(point):
            x, y = point
            point = pixels[y, x]

            mask = cv2.inRange(point, lower, upper)
            if len(np.argwhere(mask)) == 3:
                return "1"
            else:
                return "0"

        return [player_has_cards(point) for point in points]

    def count_player_cards(self, image: Image):

        upper = np.array([250, 250, 250], np.uint8)
        lower = np.array([230, 230, 230], np.uint8)
        pixels = np.asarray(image)

        x, y = 955, 961
        point = pixels[y, x]

        mask = cv2.inRange(point, lower, upper)
        if len(np.argwhere(mask)) == 3:
            return 2
        else:
            return 0

    def count_table_cards(self, image: Image):
        points = [
            (645, 537),
            (790, 537),
            (938, 537),
            (1093, 537),
            (1233, 537),
        ]

        upper = np.array([250, 250, 250], np.uint8)
        lower = np.array([230, 230, 230], np.uint8)
        pixels = np.asarray(image)

        def player_has_cards(point):
            x, y = point
            point = pixels[y, x]

            mask = cv2.inRange(point, lower, upper)
            if len(np.argwhere(mask)) == 3:
                return 1
            else:
                return 0

        return sum([player_has_cards(point) for point in points])

    def get_classification(self, image):

        player_cards = str(self.count_player_cards(image))
        table_cards = str(self.count_table_cards(image))
        opponents = self.player_card_detection(image)

        labels = [str(self.hand.dealer_pos)] + \
            [player_cards] + [table_cards] + opponents

        return ",".join(labels)

    def classify(self, plt_object, debug=False):
        image = open_image(self.file_path)

        classification = self.get_classification(image)

        if debug:
            print(self.hand.text)
            print("\n\n")
            print(classification)
            plt_object.set_data(image)
            plt.draw()
            input("")

        if not debug:
            self.save_to_dataset(image, classification)


def get_hand_histories():
    with open('poker.txt') as f:
        text = f.read()
        hands = text.split('\n\n')
        parsed = [GGPokerHandHistory(
            hand, GAME_TYPE.get_num_players()) for hand in hands[:-1]]
        sorted_parsed = list(
            sorted(parsed, key=lambda hand: hand.start_time))
        return sorted_parsed


def get_image_uuids():
    dir = Path(f"{UNCLASSIDIED_NAME}")
    files = os.listdir(dir)
    files = [f[:-4] for f in files]
    files.sort(key=lambda x: os.path.getmtime(
        os.path.join(UNCLASSIDIED_NAME, f"{x}.png")))
    return files[54:]


if __name__ == "__main__":
    _, model = model_factory(GameType.SixPlayer)

    hands = get_hand_histories()
    images = get_image_uuids()
    image_index = 0

    sucesses = 0
    failures = 0

    # open plot object
    plt_object = None
    if DEBUG:
        filename = f"{UNCLASSIDIED_NAME}/{images[1]}.png"
        plt_object = plt.imshow(open_image(filename))
        plt.pause(0.01)

    with tqdm(images) as image_uuids:
        for uuid in image_uuids:
            try:
                classification = StateClassification(uuid, hands)
                classification.classify(plt_object, debug=DEBUG)
                sucesses += 1
            except Exception as e:
                print(e)
                failures += 1

    print(f"Done! ({sucesses} sucessful, {failures} failures)\n")
    print(f"saved to {DATASET_NAME}.")
