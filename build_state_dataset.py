import torch
from data.img_transformers import poker_img_transformer
import datetime
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from data import DATASET_DIR, UN_CLASSIFIED_DIR
import os
from PIL.Image import open as open_image
from data.GGPokerHandHistory import GGPokerHandHistory
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATASET_NAME = "6_player"
UNCLASSIDIED_NAME = UN_CLASSIFIED_DIR
NUM_PLAYERS = 6


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

        tensor_img = poker_img_transformer(image)
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

    def get_classification(self):
        print(f"\n{self.hand.dealer_pos}  player_card  table_cards  num_players")
        user_input = input("->")
        labels = [str(self.hand.dealer_pos)] + user_input.split(" ")
        return ",".join(labels)

    def classify(self, plt_object):
        image = open_image(self.file_path)
        plt_object.set_data(image)
        plt.draw()

        classification = self.get_classification()
        self.save_to_dataset(image, classification)


def get_hand_histories():
    with open('poker.txt') as f:
        text = f.read()
        hands = text.split('\n\n')
        parsed = [GGPokerHandHistory(hand, NUM_PLAYERS) for hand in hands[:-1]]
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
                classification.classify(plt_object)
                sucesses += 1
            except Exception as e:
                print(e)
                failures += 1

    print(f"Done! ({sucesses} sucessful, {failures} failures)\n")
    print(f"saved to {DATASET_NAME}.")
