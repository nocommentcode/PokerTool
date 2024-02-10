from ranges.HandEvalution import HandEvalution


import numpy as np


class Pair(HandEvalution):
    multiplier = 10 ** 2
    name = "Pair"

    def __init__(self, pairs, threes, kickers):
        self.evalutate(pairs, threes, kickers)

    def evalutate(self, pairs, threes,  kickers):
        pair = (pairs != 0).sum(2) == 1
        full_house = np.any(threes != 0, axis=2)
        pair = np.logical_and(pair, np.invert(full_house))

        pair_multi = 100.0
        score = (pairs[:, :, 0] * pair_multi) + \
            (kickers[:, :, 0] + kickers[:, :, 1] + kickers[:, :, 2])
        score /= pair_multi
        score *= pair

        self.score = score
        self.hit = pair
