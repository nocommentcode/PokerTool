from ranges.HandEvalution import HandEvalution


import numpy as np


class ThreeOfAKind(HandEvalution):
    multiplier = 10 ** 6
    name = "Three of a kind"

    def __init__(self, threes, pairs, kickers):
        self.evalutate(threes, pairs, kickers)

    def evalutate(self, threes, pairs, kickers):
        three_of_a_kind = np.any(threes != 0, axis=2)
        full_house = np.any(pairs != 0, axis=2)
        three_of_a_kind = np.logical_and(
            three_of_a_kind, np.invert(full_house))
        three_multi = 10.0
        score = (threes[:, :, 0] * three_multi) + \
            (kickers[:, :, 0] + kickers[:, :, 1])
        score /= three_multi
        score *= three_of_a_kind
        self.score = score
        self.hit = three_of_a_kind
