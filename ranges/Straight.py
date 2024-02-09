import numpy as np
from ranges.HandEvalution import HandEvalution
strided = np.lib.stride_tricks.as_strided


class Straight(HandEvalution):
    multiplier = 10 ** 8
    name = "Straight"

    def __init__(self, counts) -> None:
        self.evalutate(counts)

    def evalutate(self, counts):
        # add a card at 0 to account for ace low
        add_low = counts[:, :, 0]
        straight_cards = np.append(counts, add_low[:, :, None], axis=2)
        straight_cards = straight_cards > 0

        (iter_stride, player_stride, card_stride) = straight_cards.strides
        (iter_idx, player_idx, _) = straight_cards.shape

        straights = strided(straight_cards,
                            shape=(11, iter_idx, 5, player_idx),
                            strides=(card_stride, iter_stride, card_stride, player_stride))

        straights_variants = np.all(straights, axis=2)
        straight = np.any(straights_variants, axis=0)

        highest_card = straight[..., 0, :] * \
            np.arange(13, 2, -1)[:, None, None]

        score = (highest_card * straights_variants).max(axis=0)
        self.score = score
        self.hit = straight
