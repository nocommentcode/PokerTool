from ranges.HandEvalution import HandEvalution


import numpy as np


class HighCard(HandEvalution):
    multiplier = 1
    name = "High Card"

    def __init__(self, evals, kickers) -> None:
        self.evalutate(evals, kickers)

    def evalutate(self, evals, kickers):
        card_values = np.array([1 * 10 ** (i * 2) for i in range(4, -1, -1)])

        highcard = np.invert(np.any(
            np.stack([eval.hit for eval in evals], axis=0), axis=0))
        score = (kickers * card_values).sum(2) / card_values.sum()

        self.score = score * highcard
        self.hit = highcard
