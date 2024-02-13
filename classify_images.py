import shutil
import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from build_state_dataset import UNCLASSIDIED_NAME
from data import CLASSIFIED_DIR, DATASET_DIR, UN_CLASSIFIED_DIR
import os
from data.img_transformers import cards_transformer
from PIL.Image import open as open_image
from data.GGPokerHandHistory import GGPokerHandHistory
from enums.GameType import GameType
from networks.StateDetector import StateDetector
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

STATE_DECTOR_NAME = "8_player_state_detector_2"
DIR_NAME = UN_CLASSIFIED_DIR
GAME_TYPE = GameType.EightPlayer
DEBUG = False
DATASET_NAME = "combined_6_8"


class ImageClassification:
    def __init__(self, uuid, hands):
        self.uuid = uuid
        self.file_path = os.path.join(
            DIR_NAME, f"{self.uuid}.png")

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

        tensor_img = cards_transformer(image)
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

    def get_classification(self, player_cards, table_cards):
        classification = ""

        sorted_player_cards = self.sort_player_cards_from_hand()
        for i in range(2):
            classification += ","
            if i < player_cards:
                card = sorted_player_cards[i]
                classification += f"{str(card.value)}{card.suit.to_non_symbol_string()}"

        for i in range(5):
            classification += ","
            if i < table_cards:
                card = self.hand.table_cards[i]
                classification += f"{str(card.value)}{card.suit.to_non_symbol_string()}"
        classification += ","
        classification = classification[1:]
        return classification

    def classify(self, card_detector, prev_state, plt_object, debug=True):
        image = open_image(self.file_path)

        state = card_detector.get_state(image)

        if prev_state is not None and prev_state.player_card_count == state.player_card_count and prev_state.table_card_count == state.table_card_count:
            raise Exception(
                f"Prev state == current state - {prev_state.player_card_count} {state.player_card_count}, {prev_state.table_card_count} {state.table_card_count}")

        classification = self.get_classification(
            state.player_card_count, state.table_card_count)

        if debug:
            print("\n" + classification)
            plt_object.set_data(image)
            plt.draw()
            input("")

        if not debug:
            self.save_to_dataset(image, classification)

        return state


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
    dir = Path(f"{DIR_NAME}")
    files = os.listdir(dir)
    files = [f[:-4] for f in files]
    files.sort(key=lambda x: os.path.getmtime(
        os.path.join(DIR_NAME, f"{x}.png")))
    return files


def delete_all_unclassified():
    for file in os.listdir(Path(f"{DIR_NAME}")):
        path = Path(f"{DIR_NAME}/{file}")
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (path, e))


if __name__ == "__main__":
    card_detector = StateDetector.load(
        STATE_DECTOR_NAME, game_type=GAME_TYPE)
    card_detector.eval()

    hands = get_hand_histories()
    images = get_image_uuids()

    sucesses = 0
    failures = 0
    prev_state = None

    # open plot object
    plt_object = None
    if DEBUG:
        filename = f"{UNCLASSIDIED_NAME}/{images[1]}.png"
        plt_object = plt.imshow(open_image(filename))
        plt.pause(0.01)

    with tqdm(images) as image_uuids:
        for uuid in image_uuids:
            try:
                classification = ImageClassification(uuid, hands)
                prev_state = classification.classify(
                    card_detector, prev_state, plt_object, debug=DEBUG)
                sucesses += 1
            except Exception as e:
                print(e)
                failures += 1

    print(f"Done! ({sucesses} sucessful, {failures} failures)\n")

    delete = input("delete all unclassified images? (y)")
    if delete == "":
        delete_all_unclassified()
