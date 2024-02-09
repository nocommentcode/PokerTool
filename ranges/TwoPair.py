from ranges.HandEvalution import HandEvalution


import numpy as np


class TwoPair(HandEvalution):
    multiplier = 10 ** 4
    name = "Two pair"

    def __init__(self, pairs, threes, kickers) -> None:
        self.evalutate(pairs, threes, kickers)

    def evalutate(self, pairs, threes, kickers):
        two_pair = (pairs != 0).sum(2) >= 2
        full_house = np.any(threes != 0, axis=2)
        two_pair = np.logical_and(two_pair, np.invert(full_house))

        pair_1_multi = 1000.0
        pair_2_multi = 10.0
        score = (pairs[:, :, 0] * pair_1_multi) + (pairs[:, :, 1] * pair_2_multi) + (
            kickers[:, :, 0])
        score /= pair_1_multi
        score *= two_pair

        self.hit = two_pair
        self.score = score
