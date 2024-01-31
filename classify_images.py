import shutil
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

from tqdm import tqdm

from data import CLASSIFIED_DIR, UN_CLASSIFIED_DIR
import os
from PIL.Image import open as open_image

from data.GGPokerHandHistory import GGPokerHandHistory
from enums.Value import Value
from networks.CardDetector import CardDetector
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageClassification:
    def __init__(self, uuid, hands):
        self.uuid = uuid
        self.file_path = os.path.join(
            UN_CLASSIFIED_DIR, f"{self.uuid}.png")

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
            # early = (next_hand.start_time - time_created).total_seconds() < 5
            # late = (time_created - current_hand.start_time).total_seconds() < 5
            # print(
            #     f"\n{current_hand.player_cards[0]},{current_hand.player_cards[1]}")
            # print(
            #     f"{current_hand.table_cards[0]}{current_hand.table_cards[1]}")
            # print("late" if early else "early")
            # image = open_image(self.file_path)
            # plt.imshow(image)
            # plt.show()
            raise Exception(
                f"Screenshot time ({time_created}) too close to end of hand ({next_hand.start_time }), skipping...")

        return current_hand

    def save_to_classified(self, image, classification):
        dir = Path(f"{CLASSIFIED_DIR}/{self.uuid}")
        dir.mkdir(parents=True, exist_ok=True)

        image.save(f"{dir}/image.png")

        with open(f"{dir}/classification.txt", 'w') as f:
            f.write(classification)
            f.close()

    def count_cards_using_model(self, image, card_detector):
        pred_player_count, pred_table_count = card_detector.get_card_counts(
            image)
        return pred_player_count, pred_table_count

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
        classification = f"{self.hand.num_players},{self.hand.dealer_pos}"

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

        return classification

    def classify(self, card_detector):
        image = open_image(self.file_path)
        player_cards, table_cards = self.count_cards_using_model(
            image, card_detector)

        classification = self.get_classification(player_cards, table_cards)
        # print(f"\n\n\n\n\n{classification}")
        # plt.imshow(image)
        # plt.show()
        self.save_to_classified(image, classification)


def get_hand_histories():
    with open('poker.txt') as f:
        text = f.read()
        hands = text.split('\n\n')
        parsed = [GGPokerHandHistory(hand) for hand in hands[:-1]]
        sorted_parsed = list(
            sorted(parsed, key=lambda hand: hand.start_time))
        return sorted_parsed


def get_image_uuids():
    dir = Path(f"{UN_CLASSIFIED_DIR}")
    files = os.listdir(dir)
    files = [f[:-4] for f in files]
    files.sort(key=lambda x: os.path.getmtime(
        os.path.join(UN_CLASSIFIED_DIR, f"{x}.png")))
    return files


def delete_all_unclassified():
    for file in os.listdir(Path(f"{UN_CLASSIFIED_DIR}")):
        path = Path(f"{UN_CLASSIFIED_DIR}/{file}")
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (path, e))


if __name__ == "__main__":
    card_detector = CardDetector.load("card_detector")
    card_detector.eval()

    hands = get_hand_histories()
    images = get_image_uuids()
    image_index = 0

    sucesses = 0
    failures = 0

    with tqdm(images) as image_uuids:
        for uuid in image_uuids:
            try:
                classification = ImageClassification(uuid, hands)
                classification.classify(card_detector)
                sucesses += 1
            except Exception as e:
                print(e)
                failures += 1

    print(f"Done! ({sucesses} sucessful, {failures} failures)\n")

    delete = input("delete all unclassified images? (y)")
    if delete == "":
        delete_all_unclassified()
