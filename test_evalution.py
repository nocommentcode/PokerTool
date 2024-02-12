import math
import time
import numpy as np
import numpy as np
from enums.Card import Card
from enums.GameType import GameType
from enums.Suit import Suit
from enums.Value import Value
from enums.Hand import Hand
from poker.GameState import GameState
from ranges.Evaluation import Evaluation
from ranges.Flush import Flush
from ranges.FourOfAKind import FourOfAKind
from ranges.FullHouse import FullHouse
from ranges.HighCard import HighCard
from ranges.Pair import Pair
from ranges.PostFlopEvaluation import PostFlopEvaluation
from ranges.Straight import Straight
from ranges.StraightFlush import StraightFlush
from ranges.ThreeOfAKind import ThreeOfAKind
from ranges.TwoPair import TwoPair


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
    # run_tests()
    # print("tests passed")
    hand = H("KH", "QH")
    table_cards = [C("7H"), C("TH"), C("3H"), C("6H")]
    # table_cards = []
    # opponents = [Position.SB, Position.BB, Position.SB]
    opponents = [400, 500]

    eval = Evaluation(hand, table_cards, 4)

    start = time.time()
    gs = GameState(
        GameType.EightPlayer, hand.cards(), table_cards, 0, [0, 0, 0, 0, 1, 1, 1])

    g = PostFlopEvaluation(hand, table_cards, gs)
    print(str(g))
    print(f"Took {str(time.time() - start)} s")

    # start = time.time()
    # weighted_result = eval.weighted_evaluation(opponents, 0.7)
    # print(
    #     f"Weighted tight: {weighted_result.equity} ({str(time.time() - start)})")

    # start = time.time()
    # weighted_result = eval.weighted_evaluation(opponents, debug=True)
    # print(f"Weighted: {weighted_result.equity} ({str(time.time() - start)})")

    # start = time.time()
    # weighted_result = eval.weighted_evaluation(opponents, 1.4)
    # print(
    #     f"Weighted loose: {weighted_result.equity} ({str(time.time() - start)})")

    start = time.time()
    random_result = eval.random_evaluation()
    print(f"Random: {random_result.equity} ({str(time.time() - start)})")
