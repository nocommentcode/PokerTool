from ranges.HandEvalution import HandEvalution


class FourOfAKind(HandEvalution):
    multiplier = 10 ** 14
    name = "Four of a kind"

    def __init__(self, fours, kickers):
        self.evalutate(fours, kickers)

    def evalutate(self, fours, kickers):
        four_of_a_kind = fours != 0

        # score will be value.kicker ie four 9s, 5 kicker = 9.5
        four_multi = 100
        kicker_multi = 1
        score = (fours * four_multi +
                 kickers[:, :, 0] * kicker_multi) / four_multi

        # shape = (iters, players)
        self.score = score
        self.hit = four_of_a_kind
