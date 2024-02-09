from PIL.Image import Image
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

DATASET_NAME = "9_player_state"
UNCLASSIDIED_NAME = UN_CLASSIFIED_DIR
# UNCLASSIDIED_NAME = "images/unclassified_images_from_big_batch_2"
GAME_TYPE = GameType.NinePlayer


def player_card_detection(image: Image):
    points = (
        (400, 848),
        (166, 513),
        (362, 244),
        (728, 151),
        (1080, 151),
        (1450, 230),
        (1649, 517),
        (1418, 844)
    )

    upper = np.array([150, 62, 62], np.uint8)
    lower = np.array([112, 40, 40], np.uint8)
    pixels = np.asarray(image)

    def player_has_cards(index):
        x, y = points[index]
        point = pixels[y, x]
        print(point)

        mask = cv2.inRange(point, lower, upper)
        if len(np.argwhere(mask)) == 3:
            return "1"
        else:
            return "0"

    return [player_has_cards(i) for i in range(8)]


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

    def get_classification(self, predictions, image):

        card, table = predictions

        player_cards = str(card)
        table_cards = str(table)
        opponents = player_card_detection(image)

        # opponents = ["0" if i not in opponents else "1" for i in range(
        # 1, GAME_TYPE.get_num_players())]
        labels = [str(self.hand.dealer_pos)] + \
            [player_cards] + [table_cards] + opponents

        return ",".join(labels)
    # if predictions is None:
        #     print(
        #         f"\n{self.hand.dealer_pos}  player_card  table_cards  num_players")
        #     user_input = input("->")
        #     user_input = user_input.split(" ")
        #     player_cards = user_input[0]
        #     table_cards = user_input[1]
        #     opponents = [int(char) for char in user_input[2]]

        # else:
        #     card, table = predictions
        #     print(
        #         f"\n{self.hand.dealer_pos}  {card}  {table}  num_players (empty to change)")
        #     user_input = input("->")

        #     # change predictions
        #     if user_input == "":
        #         return self.get_classification(None)

        #     player_cards = str(card)
        #     table_cards = str(table)
        #     opponents = [int(char) for char in user_input]

        # opponents = ["0" if i not in opponents else "1" for i in range(
        #     1, GAME_TYPE.get_num_players())]
        # labels = [str(self.hand.dealer_pos)] + \
        #     [player_cards] + [table_cards] + opponents

        # return ",".join(labels)

    def pred_counts_via_model(self, image, poker_network: PokerNetwork):
        transformed = cards_transformer(image)
        batch = transformed.unsqueeze(0)
        batch = batch.to(torch.float32).to("cuda")

        predictions = poker_network.predict(batch)
        card_count = 0
        for type in PLAYER_CARDS:
            if predictions[type.value].suit != Suit.Empty:
                card_count += 1

        table_count = 0
        for type in TABLE_CARDS:
            if predictions[type.value].suit != Suit.Empty:
                table_count += 1

        return card_count, table_count

    def classify(self, plt_object, poker_network=None):
        image = open_image(self.file_path)
        # plt_object.set_data(image)
        # plt.draw()
        # plt.imshow(image)

        predictions = None
        if poker_network is not None:
            predictions = self.pred_counts_via_model(image, poker_network)

        classification = self.get_classification(predictions, image)
        # print(classification)
        # plt.show()
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
    filename = f"{UNCLASSIDIED_NAME}/{images[1]}.png"
    plt_object = plt.imshow(open_image(filename))
    plt.pause(0.01)

    with tqdm(images) as image_uuids:
        for uuid in image_uuids:
            try:
                classification = StateClassification(uuid, hands)
                classification.classify(plt_object, model)
                sucesses += 1
            except Exception as e:
                print(e)
                failures += 1

    print(f"Done! ({sucesses} sucessful, {failures} failures)\n")
    print(f"saved to {DATASET_NAME}.")
