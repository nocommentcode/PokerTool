import time
import numpy as np


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

    def get_hands_still_in_deck(self):
        hands = np.load("hand_strengths.npy")
        mask_card_1 = np.isin(hands[:, 0], self.deck)
        mask_card_2 = np.isin(hands[:, 1], self.deck)

        mask = np.logical_and(mask_card_1, mask_card_2)
        hands = hands[mask]

        return hands

    def get_opponent_hand_idx(self, strength, used_indexes):
        def generate(count):
            return np.random.randint(low=0,
                                     high=strength,
                                     size=(count))

        def get_duplicates(hand_idxes):
            num_used = used_indexes.shape[1]
            equal_dims = []
            for i in range(num_used):
                equal_dims.append(hand_idxes == used_indexes[:, i])
            equal_dims = np.array(equal_dims)
            return np.logical_or.reduce(equal_dims)

        hand_idxes = generate(self.num)
        duplicate_idx = get_duplicates(hand_idxes)
        while np.any(duplicate_idx):
            indicies, = np.where(duplicate_idx)
            hand_idxes[indicies] = generate(len(indicies))
            duplicate_idx = get_duplicates(hand_idxes)

        return hand_idxes

    def weighted_shuffle(self, hand_strengths):
        hands = self.get_hands_still_in_deck()

        opponent_hands = np.zeros((self.num, 2, 0))
        used_indexes = np.zeros((self.num, 0))

        def get_hand_indicies_with_clashes(opponent_hand, hands):
            card1 = np.logical_or(
                opponent_hand[:, 0][:, None] == hands[:, 0],
                opponent_hand[:, 0][:, None] == hands[:, 1],
            )
            card2 = np.logical_or(
                opponent_hand[:, 1][:, None] == hands[:, 0],
                opponent_hand[:, 1][:, None] == hands[:, 1],
            )

            both_cards = np.logical_or(card1, card2)
            indicies = np.where(both_cards)[1].reshape((self.num, -1))

            return indicies

        for strength in hand_strengths:
            start = time.time()

            opponent_idx = self.get_opponent_hand_idx(strength, used_indexes)
            opponent_hand = hands[opponent_idx]
            opponent_hands = np.append(
                opponent_hand[:, :, None], opponent_hands, 2)
            print(f"opponent cards : {str(time.time()- start)}")

            # remove indicies from hands
            start = time.time()
            clashing_indicies = get_hand_indicies_with_clashes(
                opponent_hand, hands)
            print(f"getting indicies : {str(time.time()- start)}")

            start = time.time()

            used_indexes = np.append(
                used_indexes, clashing_indicies,  1)
            print(f"appending : {str(time.time()- start)}")

            start = time.time()

            # remove cards from deck
            opponent_hand_mask = np.logical_or(opponent_hand[:, 0][:, None] == self.deck,
                                               opponent_hand[:, 1][:, None] == self.deck)
            deck_mask = np.invert(opponent_hand_mask)
            deck = self.deck[deck_mask]
            self.deck = deck.reshape((self.num, -1))
            print(f"remove from deck : {str(time.time()- start)}")

        self.random_shuffle(len(hand_strengths))

        opponent_hands = opponent_hands.reshape((self.num, -1))
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
        self.table_start_idx += num_cards
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
    deck = Deck(10000)
    cards_to_remove = np.array([7, 52, 50, 43])
    deck.remove_cards(cards_to_remove)

    # hands = deck.get_hands_still_in_deck()
    # print(hands.shape)
    # for hand in hands:
    #     if hand[0] in cards_to_remove or hand[1] in cards_to_remove:
    #         print(f"Error {hand}")

    deck.weighted_shuffle([100, 250, 300])

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


if __name__ == "__main__":
    start = time.time()
    deck = Deck(10000)
    cards_to_remove = np.array([7, 52, 50, 43])
    deck.remove_cards(cards_to_remove)
    deck.random_shuffle(3)
    print(f"Random: {str(time.time() - start)}")

    # test_weighted_shuffle()
    start = time.time()
    deck = Deck(10000)
    cards_to_remove = np.array([7, 52, 50, 43])
    deck.remove_cards(cards_to_remove)
    deck.weighted_shuffle([250, 170, 320])
    print(f"Weighted: {str(time.time() - start)}")
