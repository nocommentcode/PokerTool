from ranges.HandEvalution import HandEvalution


import numpy as np


class Flush(HandEvalution):
    multiplier = 10 ** 10
    name = "Flush"

    def __init__(self, cards, suits):
        self.evalutate(cards, suits)

    def evalutate(self, cards, suits):
        iterations, _, players = suits.shape
        suit_counts = (suits[:, :, :, None] == np.arange(4)).sum(1)
        max_suit = np.argmax(suit_counts, axis=2)
        flush = suit_counts[np.arange(
            iterations)[:, None], np.arange(players), max_suit] >= 5

        flush_cards = suits == max_suit[:, None]
        sorted_flushcards = np.sort(flush_cards * cards, axis=1)[:, ::-1]
        sorted_flushcards = sorted_flushcards[:, :5]

        multiplier = np.array([1 * 10 ** (i * 2) for i in range(4, -1, -1)])

        score = (sorted_flushcards *
                 multiplier[..., None]).sum(1) * flush / np.sum(multiplier[:])
        self.score = score
        self.hit = flush
