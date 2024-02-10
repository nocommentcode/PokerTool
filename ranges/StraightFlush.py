from ranges.HandEvalution import HandEvalution


import numpy as np
strided = np.lib.stride_tricks.as_strided


class StraightFlush(HandEvalution):
    multiplier = 10 ** 18
    name = "Straight Flush"

    def __init__(self, cards, suits) -> None:
        self.evalutate(cards, suits)

    def evalutate(self, cards, suits):
        # card_suits = (suit_idx x iterations x card_idx x player)
        card_suits = (suits == np.arange(4)[:, None, None, None]) * cards
        card_suits = np.sort(card_suits, axis=2)[:, :, ::-1]

        # add a card at 0 to account for ace low
        add_low = card_suits[:, :, 0] - 13
        card_suits = np.append(card_suits, add_low[:, :, None, :], axis=2)

        # build array of different possible straights for each iter, player and suit
        (suit_stride, iter_stride, card_stride, player_stride) = card_suits.strides
        (_, iter_idx, _, player_idx) = card_suits.shape
        # straight_idx x suit x iterations x 5 cards x player
        straights = strided(card_suits,
                            shape=(4, 4, iter_idx, 5, player_idx),
                            strides=(card_stride, suit_stride, iter_stride, card_stride, player_stride))

        # find highest card in each scenario
        highest_card = np.sum(straights[..., 0, :], axis=0)

        # find scenarios where is a flush and a straight
        all_straight_flushes = np.all(np.diff(straights, axis=3) == -1, axis=3)
        straight_flushes_by_suit = np.any(all_straight_flushes, axis=0)

        # shape = (iters x players)
        self.score = np.sum(highest_card * straight_flushes_by_suit, axis=0)
        self.hit = np.any(straight_flushes_by_suit, axis=0)
