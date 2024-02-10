import math
import os
import time
import numpy as np
from enums.Card import Card

from enums.Suit import Suit
from enums.Value import Value

BASE_STARTING_HAND_DIR = "starting_hands"

STARTING_HAND_FILENAME = "starting_hands.npy"
STARTING_HAND_CLASHES_FILENAME = "starting_hands_clashes.npy"


class Deck:
    def __init__(self, num):
        self.num = num
        self.shuffled = False

        deck = np.arange(1, 53)
        self.deck = np.tile(deck, reps=(num, 1))

        self.table_start_idx = None

    def random_shuffle(self, num_players):
        temp_random = np.random.random(self.deck.shape)
        idx = np.argsort(temp_random, axis=-1,)
        self.deck = self.deck[np.arange(self.deck.shape[0])[:, None], idx]
        self.shuffled = True
        self.table_start_idx = num_players * 2

    # def num_to_card(self, num: int):
    #     value, suit = math.ceil(num / 4), num % 4
    #     if value == 13:
    #         value = 1
    #     else:
    #         value += 1
    #     return Card(Suit.from_index(int(suit + 1)),
    #                 Value.from_index(int(value)))

    def get_hands_still_in_deck(self):
        hands = np.load(os.path.join(
            BASE_STARTING_HAND_DIR, STARTING_HAND_FILENAME))

        # get indexes used by cards already delt
        mask_card_1 = np.isin(hands[:, 0], self.deck, invert=True)
        mask_card_2 = np.isin(hands[:, 1], self.deck, invert=True)
        mask = np.logical_or(mask_card_1, mask_card_2)
        delt_hand_indexes = np.where(mask)[0]

        return hands, delt_hand_indexes

    def get_opponent_hand_idx(self, probs, used_indexes, delt_hand_indexes):
        def generate(count):
            return np.random.choice(len(probs),
                                    p=probs,
                                    size=count)

        def get_duplicates(hand_idxes):
            num_used = used_indexes.shape[1]
            equal_dims = []
            for i in range(num_used):
                equal_dims.append(hand_idxes == used_indexes[:, i])

            equal_dims.append(
                np.any(hand_idxes == delt_hand_indexes[:, None], 0))

            equal_dims = np.array(equal_dims)
            return np.logical_or.reduce(equal_dims)

        hand_idxes = generate(self.num)
        duplicate_idx = get_duplicates(hand_idxes)
        while np.any(duplicate_idx):
            indicies, = np.where(duplicate_idx)
            hand_idxes[indicies] = generate(len(indicies))
            duplicate_idx = get_duplicates(hand_idxes)

        return hand_idxes

    def weighted_shuffle(self, hand_probabilities):
        hands, delt_hand_indexes = self.get_hands_still_in_deck()
        hand_clashes = np.load(os.path.join(
            BASE_STARTING_HAND_DIR, STARTING_HAND_CLASHES_FILENAME))

        opponent_hands = np.zeros((self.num, 2, 0))
        used_indexes = np.zeros((self.num, 0))
        for probs in hand_probabilities:
            opponent_idx = self.get_opponent_hand_idx(
                probs, used_indexes, delt_hand_indexes)
            opponent_hand = hands[opponent_idx]
            opponent_hands = np.append(opponent_hands,
                                       opponent_hand[:, :, None], 2)

            # remove indicies from hands
            clashing_indicies = hand_clashes[opponent_idx]
            used_indexes = np.append(
                used_indexes, clashing_indicies,  1)

            # remove cards from deck
            opponent_hand_mask = np.logical_or(opponent_hand[:, 0][:, None] == self.deck,
                                               opponent_hand[:, 1][:, None] == self.deck)
            deck_mask = np.invert(opponent_hand_mask)
            deck = self.deck[deck_mask]
            self.deck = deck.reshape((self.num, -1))

        self.random_shuffle(len(hand_probabilities) + 1)

        opponent_hands = opponent_hands.reshape((self.num, -1), order="F")
        self.deck = np.append(opponent_hands, self.deck, 1)

    def remove_cards(self, cards):
        if self.shuffled:
            raise Exception(
                "Remove cards before shuffling deck, deck is already shuffled")

        mask = np.isin(self.deck, cards, invert=True)
        self.deck = self.deck[mask]
        self.deck = self.deck.reshape((self.num, -1))

    def deal_table_cards(self, num_cards):
        if not self.shuffled:
            raise Exception("Deck is not shuffled")

        idx = self.table_start_idx
        return self.deck[:, idx:idx+num_cards]

    def deal_player_cards(self, player_index):
        # returns num x 2
        if not self.shuffled:
            raise Exception("Deck is not shuffled")

        idx = player_index * 2
        return self.deck[:, idx:idx+2]


def test_remove_cards():
    deck = Deck(10)
    cards_to_remove = np.array([7, 52, 50, 43])
    deck.remove_cards(cards_to_remove)
    print(deck.deck.shape)
    print(deck.deck[0])
    print(deck.deck[1])
    print(deck.deck[2])


def test_random_shuffle():
    deck = Deck(10)
    deck.random_shuffle(3)

    print("sorted:")
    print(np.sort(deck.deck[0]))

    print("unsorted")
    print(deck.deck[0])

    print("Table cards:")
    print(deck.deal_table_cards(5)[0])

    print("Player 1 cards:")
    print(deck.deal_player_cards(0)[0])

    print("Player 2 cards:")
    print(deck.deal_player_cards(1)[0])

    print("Player 3 cards:")
    print(deck.deal_player_cards(2)[0])


def test_weighted_shuffle():
    # np.random.seed(0)
    deck = Deck(10)
    cards_to_remove = np.array([7, 52, 50, 43])
    deck.remove_cards(cards_to_remove)

    hands = deck.get_hands_still_in_deck()
    print(hands.shape)
    for hand in hands:
        if hand[0] in cards_to_remove or hand[1] in cards_to_remove:
            print(f"Error {hand}")

    deck.weighted_shuffle([2, 7, 10])

    for i in range(3):
        print("sorted:")
        print(np.sort(deck.deck[i]))

        print("unsorted")
        print(deck.deck[i])

        print("Table cards:")
        print(deck.deal_table_cards(5)[i])

        print("Player 1 cards:")
        print(deck.deal_player_cards(0)[i])

        print("Player 2 cards:")
        print(deck.deal_player_cards(1)[i])

        print("Player 3 cards:")
        print(deck.deal_player_cards(2)[i])


if __name__ == "__main__":
    # start = time.time()
    # deck = Deck(10000)
    # cards_to_remove = np.array([7, 52, 50, 43])
    # deck.remove_cards(cards_to_remove)
    # deck.random_shuffle(3)
    # print(f"Random: {str(time.time() - start)}")

    test_weighted_shuffle()
    # start = time.time()
    # deck = Deck(1000)
    # cards_to_remove = np.array([7, 52, 50, 43])
    # deck.remove_cards(cards_to_remove)
    # deck.weighted_shuffle([250, 170, 320])
    # print(f"Weighted: {str(time.time() - start)}")
