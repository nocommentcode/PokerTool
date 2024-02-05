from enum import Enum
import time
from typing import List
import numpy as np


strided = np.lib.stride_tricks.as_strided


class Suit(Enum):
    Empty = 0
    Spades = 1
    Hearts = 2
    Diamonds = 3
    Clubs = 4

    @staticmethod
    def from_string(string: str) -> "Suit":
        if string == 'S':
            return Suit.Spades

        if string == 'D':
            return Suit.Diamonds

        if string == 'H':
            return Suit.Hearts

        if string == 'C':
            return Suit.Clubs

        raise AttributeError(f"Suit {string} does not exist")

    @staticmethod
    def from_index(index: int):
        suits = [Suit.Empty, Suit.Spades,
                 Suit.Hearts, Suit.Diamonds, Suit.Clubs]
        return suits[index]

    def __str__(self):
        symbols = ['', '♠', '♥', '♦', '♣']
        return symbols[self.value]

    def __eq__(self, other):
        if type(other) != Suit:
            return False

        return self.value == other.value

    def to_non_symbol_string(self):
        if self.value == 0:
            return ''

        if self.value == 1:
            return "S"

        if self.value == 2:
            return "H"

        if self.value == 3:
            return "D"

        if self.value == 4:
            return "C"

    def __gt__(self, other):
        if type(other) != Suit:
            return False
        # spade > club
        # heart > club
        # diamond > club
        # diamond > heart
        # spade > heart
        # spade > diamond
        suit_order = [Suit.Clubs.value,
                      Suit.Hearts.value,
                      Suit.Diamonds.value,
                      Suit.Spades.value]
        my_index = suit_order.index(self.value)
        other_index = suit_order.index(other.value)
        return my_index > other_index


class Value(Enum):
    Empty = 0
    Ace = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Jack = 11
    Queen = 12
    King = 13

    @staticmethod
    def from_string(string: str) -> "Value":
        values = {"": None,
                  "A": Value.Ace,
                  "2": Value.Two,
                  "3": Value.Three,
                  "4": Value.Four,
                  "5": Value.Five,
                  "6": Value.Six,
                  "7": Value.Seven,
                  "8": Value.Eight,
                  "9": Value.Nine,
                  "T": Value.Ten,
                  "J": Value.Jack,
                  "Q": Value.Queen,
                  "K": Value.King}

        return values[string]

    @staticmethod
    def from_index(index: int):
        values = [Value.Empty,
                  Value.Ace,
                  Value.Two,
                  Value.Three,
                  Value.Four,
                  Value.Five,
                  Value.Six,
                  Value.Seven,
                  Value.Eight,
                  Value.Nine,
                  Value.Ten,
                  Value.Jack,
                  Value.Queen,
                  Value.King]

        return values[index]

    def __str__(self):
        symbols = [
            "",
            "A",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "T",
            "J",
            "Q",
            "K"
        ]
        return symbols[self.value]

    def __eq__(self, other):
        if type(other) != Value:
            return False

        return self.value == other.value

    def __gt__(self, other):
        if type(other) != Value:
            return False

        # handle aces
        if self.value == 1 and other.value != 1:
            return True
        if self.value != 1 and other.value == 1:
            return False

        return self.value > other.value


class Card:
    def __init__(self, suit: Suit, value: Value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value}{self.suit}"

    def __eq__(self, other):
        if type(other) != Card:
            return False

        return self.suit == other.suit and self.value == other.value


class Hand:
    def __init__(self, card_1: Card, card_2: Card):
        self.card_1 = card_1
        self.card_2 = card_2

    def cards(self):
        return self.card_1, self.card_2

    def is_suited(self):
        return self.card_1.suit == self.card_2.suit

    def is_pokets(self):
        return self.card_1.value == self.card_2.value

    def __str__(self) -> str:
        return f"{str(self.card_1)} {str(self.card_2)}"


class HandEvalution:
    def get_equity(self):
        return self.hit[:, 0].sum() / self.hit.shape[0]

    def __str__(self):
        return f"{self.name}: {self.get_equity()}"


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


class EvalutionResult:
    def __init__(self, evalutions, equity):
        self.evalutions = evalutions
        self.equity = equity

    def __str__(self):
        string = f"Equity: {self.equity}\n"
        for eval in self.evalutions:
            if eval.get_equity() > 0:
                string += f"{str(eval)}\n"

        return string


class Evaluation:
    def __init__(self, hand: Hand, table_cards: List[Card], player_count: int):
        self.player_cards = np.array([
            self.card_to_num(card) for card in hand.cards()])
        self.table_cards = np.array(
            [self.card_to_num(card) for card in table_cards])

        self.player_count = player_count

    @staticmethod
    def card_to_num(card: Card):
        def handle_card_value(value):
            # shift 1 down (2 = 1 in Card, should be 1 for this)
            int_value = value.value - 1
            # handle ace
            if int_value == 0:
                int_value = 13

            return int_value

        card_num, card_suit = handle_card_value(
            card.value), card.suit.value - 1

        # suit 0 -> 1 - 13
        if card_suit == 0:
            return card_num * 4

        # suit 1 -> 14 - 26
        # suit 2 -> 27 - 39
        # suit 3 -> 40 - 52
        else:
            return (card_num - 1) * 4 + card_suit

    def deal_cards(self, iterations):
        def remove_cards_from_deck(cards, deck):
            mask = np.isin(deck, cards, invert=True)
            return deck[mask]

        def shuffle_deck(deck):
            # all_decks = (iterations x num_cards)
            all_decks = np.tile(deck, reps=(iterations, 1))
            temp_random = np.random.random(all_decks.shape)
            idx = np.argsort(temp_random, axis=-1)
            return all_decks[np.arange(all_decks.shape[0])[:, None], idx]

        # 52 cards starting at 5 so that later code will make arrays starting at 2
        deck = np.arange(5, 57)
        deck = remove_cards_from_deck(self.player_cards, deck)
        deck = remove_cards_from_deck(self.table_cards, deck)
        deck = shuffle_deck(deck)

        player_hands = np.zeros((iterations, 7, self.player_count))
        cards_to_deal = 5 - len(self.table_cards)
        for i in range(self.player_count):

            # player hole cards
            if i == 0:
                player_hands[:, :2, i] = self.player_cards

            # opponent hole cards
            else:
                start_idx = cards_to_deal + (i * 2)
                end_idx = start_idx + 2
                player_hands[:, :2, i] = deck[:, start_idx: end_idx]

            # table cards already delt
            player_hands[:, 2:2+len(self.table_cards), i] = self.table_cards

            # random remaining table cards
            player_hands[:, 2+len(self.table_cards):,
                         i] = deck[:, :cards_to_deal]

        cards = np.ceil(player_hands / 4)
        suits = player_hands % 4 * 1

        # shape = (iterations, card, player)
        return cards, suits

    @staticmethod
    def get_card_counts(player_cards):
        # shape = (iters x players x card)
        return (np.arange(13, 0, -1) == player_cards[:, :, :, None]).sum(1)

    @staticmethod
    def get_of_a_kinds(card_counts):
        cards_13_to_1 = np.arange(13, 0, -1)

        # shape = (iters x player x pair_idx)
        all_pairs = np.sort((card_counts == 2) * cards_13_to_1, axis=2)
        pairs = all_pairs[:, :, ::-1][:, :, :3]

        # shape = (iters x player x three_idx)
        all_threes = np.sort((card_counts == 3) * cards_13_to_1, axis=2)
        threes = all_threes[:, :, ::-1][:, :, :2]

        # shape = (iters x players x 1)
        all_fours = np.sort((card_counts == 4) * cards_13_to_1, axis=2)
        fours = all_fours[:, :, ::-1][:, :, 0]

        # shape = (iters x players x 5)
        all_kickers = np.sort((card_counts == 1) * cards_13_to_1, axis=2)
        kickers = all_kickers[:, :, ::-1][:, :, :5]

        return pairs, threes, fours, kickers

    def run_evaluation(self, iterations=10000):
        cards, suits = self.deal_cards(iterations)

        card_counts = self.get_card_counts(cards)
        pairs, threes, fours, kickers = self.get_of_a_kinds(card_counts)

        evaluations = [StraightFlush(cards, suits),
                       FourOfAKind(fours, kickers),
                       FullHouse(pairs, threes),
                       Flush(cards, suits),
                       Straight(card_counts),
                       ThreeOfAKind(threes, pairs, kickers),
                       TwoPair(pairs, threes, kickers),
                       Pair(pairs, threes, kickers)]
        evaluations.append(HighCard(evaluations, kickers))

        equity = self.get_equity(evaluations)
        return EvalutionResult(evaluations, equity)

    def get_equity(self, evalutions):
        multipliers = np.array([eval.multiplier for eval in evalutions])
        hits = np.stack([eval.hit for eval in evalutions], axis=0)
        scores = np.stack([eval.score for eval in evalutions], axis=0)

        final_scores = multipliers[:, None, None] * hits * scores
        final_scores = np.sort(final_scores, axis=0)[::-1, :, :]

        winners = (final_scores[0, ::] == np.amax(
            final_scores[0, :, :], axis=1)[:, None])

        player_winner_mask = np.zeros(self.player_count, dtype=int)
        player_winner_mask[0] = 1
        player_wins = (winners == player_winner_mask).all(1)
        player_wins = np.sum(player_wins, axis=0)
        return player_wins / hits.shape[1]


def C(card):
    value = card[0]
    suit = card[1]
    return Card(Suit.from_string(suit), Value.from_string(value))


def H(card1, card2):
    return Hand(C(card1), C(card2))


def parse_hands(hands):
    player_hands = np.array([[[Evaluation.card_to_num(C(card))
                               for card in hand] for hand in hands]])
    player_hands = player_hands.swapaxes(1, 2)

    cards = np.ceil(player_hands / 4)
    suits = player_hands % 4 * 1
    counts = Evaluation.get_card_counts(cards)
    pairs, threes, fours, kickers = Evaluation.get_of_a_kinds(counts)

    return dict(pairs=pairs, threes=threes, fours=fours, kickers=kickers, suits=suits, cards=cards, counts=counts)


def run_test(hands, args, hits, score_order, eval_class):
    args_dict = parse_hands(hands)
    hand = eval_class(*[args_dict[arg] for arg in args])

    # check hits
    hits = np.array(hits)
    correct = hits == hand.hit[0]
    assert np.all(
        correct), f"Hit error at index: {', '.join([str(i[0]) for i in np.argwhere(np.invert(correct))])}"

    # check scores
    scores = hand.score[0]
    score_idx = np.argsort(scores*-1)[:len(score_order)]
    score_order = np.array(score_order)
    correct = score_order == score_idx
    # print(hand.score[0], np.argsort(scores*-1))
    assert np.all(
        correct), f"Score error at index: {', '.join([str(i[0]) for i in np.argwhere(np.invert(correct))])}"


def test_straight_flush():
    hands = [
        ("AS", "KS", "QS", "JS", "TS", "6D", "8D"),
        ("AS", "3S", "QS", "JS", "TS", "6D", "8D"),
        ("KD", "QD", "JD", "TD", "9D", "6D", "8D"),
        ("AD", "2D", "3D", "4D", "5D", "9D", "8D"),
        ("8H", "6D", "TC", "6C", "4H", "9D", "2S")
    ]
    hits = [True, False, True, True, False]
    score_order = [0, 2, 3]
    run_test(hands, ["cards", "suits"], hits, score_order, StraightFlush)


def test_four_of_a_kind():
    hands = [
        ("AS", "AC", "AH", "AD", "KS", "6D", "8D"),
        ("AS", "AC", "AH", "AD", "TS", "6D", "8D"),
        ("QD", "QS", "QH", "QD", "TS", "6D", "8D"),
        ("KD", "QD", "JD", "TD", "9D", "6D", "8D"),
        ("8H", "6D", "TC", "6C", "4H", "9D", "2S")
    ]
    hits = [True, True, True, False, False]
    score_order = [0, 1, 2]
    run_test(hands, ["fours", "kickers"], hits, score_order, FourOfAKind)


def test_full_house():
    hands = [
        ("AS", "AC", "KH", "KD", "KS", "6D", "8D"),
        ("KS", "KC", "KD", "AD", "AS", "6D", "8D"),
        ("KS", "KC", "KD", "QD", "QS", "6D", "8D"),
        ("AS", "AC", "AH", "AD", "KS", "6D", "8D"),
        ("8H", "8D", "TC", "TH", "5D", "5C", "2S")
    ]
    hits = [True, True, True, False, False]
    score_order = [0, 1, 2]
    run_test(hands, ["threes", "pairs"], hits, score_order, FullHouse)


def test_flush():
    hands = [
        ("AS", "KS", "QS", "TS", "4S", "6D", "8D"),
        ("QD", "3D", "3S", "7D", "AD", "6D", "8D"),
        ("KD", "3D", "3S", "7D", "2D", "6D", "8D"),
        ("AS", "AC", "AH", "AD", "KS", "6D", "8D"),
        ("8H", "8D", "TC", "TH", "5D", "5C", "2S")
    ]
    hits = [True, True, True, False, False]
    score_order = [0, 1, 2]
    run_test(hands, ["cards", "suits"], hits, score_order, Flush)


def test_straight():
    hands = [
        ("AD", "KS", "QC", "JS", "TS", "6D", "8D"),
        ("AH", "3S", "QS", "JS", "TS", "6D", "8D"),
        ("KD", "QD", "JD", "TH", "9D", "6D", "8D"),
        ("AD", "2D", "3C", "4D", "5D", "9D", "AD"),
        ("6D", "2D", "3C", "4D", "5D", "9D", "8D"),
        ("8H", "6D", "TC", "6C", "4H", "9D", "2S")
    ]
    hits = [True, False, True, True, True, False]
    score_order = [0, 2, 4, 3]
    run_test(hands, ["counts"], hits, score_order, Straight)


def test_three_of_a_kind():
    hands = [
        ("AD", "AS", "AC", "QS", "TS", "6D", "8D"),
        ("AD", "AS", "AC", "JS", "TS", "6D", "8D"),
        ("KD", "KS", "KC", "JS", "TS", "6D", "8D"),
        ("AD", "AS", "AC", "QS", "QD", "6D", "8D"),
        ("AD", "AS", "7C", "JS", "TS", "6D", "8D"),
    ]
    hits = [True, True, True, False, False]
    score_order = [0, 1, 2]
    run_test(hands, ["threes", "pairs", "kickers"],
             hits, score_order, ThreeOfAKind)


def test_two_pair():
    hands = [
        ("AD", "AS", "KC", "KS", "QS", "6D", "8D"),
        ("AD", "AS", "KC", "KS", "JS", "6D", "8D"),
        ("AD", "AS", "QC", "QS", "TS", "6D", "8D"),
        ("AD", "AS", "QC", "QS", "TS", "TD", "8D"),
        ("AD", "AS", "QC", "QS", "QD", "TD", "8D"),
        ("AD", "AS", "QC", "QS", "QD", "TD", "TD"),
    ]
    hits = [True, True, True, True, False, False]
    score_order = [0, 1, 2, 3]
    run_test(hands, ["pairs", "threes", "kickers"],
             hits, score_order, TwoPair)


def test_pair():
    hands = [
        ("AD", "AS", "KC", "QS", "TS", "6D", "8D"),
        ("AD", "AS", "QC", "TS", "JS", "6D", "8D"),
        ("AD", "AS", "KC", "KS", "KS", "6D", "8D"),
        ("AD", "AS", "9C", "KS", "KS", "6D", "8D"),
        ("AD", "JS", "QC", "7S", "2D", "TD", "8D"),
    ]
    hits = [True, True, False, False, False]
    score_order = [0, 1]
    run_test(hands, ["pairs", "threes", "kickers"],
             hits, score_order, Pair)


def test_high_card():
    hands = [
        ("AD", "7S", "KC", "QS", "TS", "6D", "8D"),
        ("2D", "7S", "KC", "QS", "TS", "6D", "8D"),
        ("AS", "KS", "QS", "JS", "TS", "6D", "8D"),
        ("AS", "AC", "AH", "AD", "KS", "6D", "8D"),
        ("AS", "AC", "KH", "KD", "KS", "6D", "8D"),
        ("AS", "KS", "QS", "TS", "4S", "6D", "8D"),
        ("AD", "KS", "QC", "JS", "TS", "6D", "8D"),
        ("AD", "AS", "AC", "QS", "TS", "6D", "8D"),
        ("AD", "AS", "KC", "KS", "QS", "6D", "8D"),
        ("AD", "AS", "KC", "QS", "TS", "6D", "8D"),
    ]
    hits = [True, True, False, False, False, False, False, False, False, False]
    score_order = [0, 1]

    player_hands = np.array([[[Evaluation.card_to_num(C(card))
                               for card in hand] for hand in hands]])
    player_hands = player_hands.swapaxes(1, 2)

    cards = np.ceil(player_hands / 4)
    suits = player_hands % 4 * 1
    card_counts = Evaluation.get_card_counts(cards)
    pairs, threes, fours, kickers = Evaluation.get_of_a_kinds(card_counts)
    evaluations = [StraightFlush(cards, suits),
                   FourOfAKind(fours, kickers),
                   FullHouse(pairs, threes),
                   Flush(cards, suits),
                   Straight(card_counts),
                   ThreeOfAKind(threes, pairs, kickers),
                   TwoPair(pairs, threes, kickers),
                   Pair(pairs, threes, kickers)]
    hand = HighCard(evaluations, kickers)

    # check hits
    hits = np.array(hits)
    correct = hits == hand.hit[0]
    assert np.all(
        correct), f"Hit error at index: {', '.join([str(i[0]) for i in np.argwhere(np.invert(correct))])}"

    # check scores
    scores = hand.score[0]
    score_idx = np.argsort(scores*-1)[:len(score_order)]
    score_order = np.array(score_order)
    correct = score_order == score_idx

    assert np.all(
        correct), f"Score error at index: {', '.join([str(i[0]) for i in np.argwhere(np.invert(correct))])}"


def run_tests():
    test_straight_flush()
    test_four_of_a_kind()
    test_full_house()
    test_flush()
    test_straight()
    test_three_of_a_kind()
    test_two_pair()
    test_pair()
    test_high_card()


if __name__ == "__main__":
    np.random.seed(8)
    player_hand = H("QS", "KS")
    table_cards = [C("5S"), C("7S"), C("TS")]
    table_cards = []
    E = Evaluation(player_hand, table_cards, 3)
    result = E.run_evaluation(iterations=2)
    print(result.evalutions[3].hit[:])
    print(result)
