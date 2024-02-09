from ranges.HandEvalution import HandEvalution


import numpy as np


class FullHouse(HandEvalution):
    multiplier = 10 ** 12
    name = "Full house"

    def __init__(self, pairs, threes) -> None:
        self.evalutate(pairs, threes)

    def evalutate(self, pairs, threes):
        three_of_a_kinds_count = (threes != 0).sum(-1)
        pairs_count = (pairs != 0).sum(-1)

        fullhouse = np.logical_or(three_of_a_kinds_count == 2, np.logical_and(
            three_of_a_kinds_count == 1, pairs_count > 0))

        three_card = threes[:, :, 0]
        two_card = np.amax(
            np.stack((pairs[:, :, 0], threes[:, :, 1]), axis=1), axis=1)

        fullHouseMulti = 100
        score = (three_card * fullHouseMulti + two_card) / fullHouseMulti

        self.score = score
        self.hit = fullhouse
