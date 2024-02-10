import math
import numpy as np
import numpy as np
from enums.Card import Card
from enums.GameType import GameType
from enums.OpponentAction import OpponentAction
from enums.Position import GAME_TYPE_POSITIONS, Position
from enums.Suit import Suit
from enums.Value import Value
from enums.Hand import Hand
from ranges import BASE_STARTING_HAND_DIR, STARTING_HAND_CLASHES_FILENAME, STARTING_HAND_FILENAME, STARTING_HAND_PROBS
from ranges.Evaluation import Evaluation
import os

from ranges.RangeChart import load_range_charts

EVALUATION_ITERATIONS = 10000
GAME_TYPE = GameType.SixPlayer


def calculate_starting_hand_strengths():
    strengths = []
    hands = []
    for suit1 in [1, 2, 3, 4]:
        for val1 in range(1, 14):
            card1 = Card(Suit.from_index(suit1),
                         Value.from_index(val1))

            for suit2 in [1, 2, 3, 4]:
                for val2 in range(1, 14):
                    card2 = Card(Suit.from_index(suit2),
                                 Value.from_index(val2))

                    if card1 == card2:
                        continue

                    num1 = Evaluation.card_to_num(card1)
                    num2 = Evaluation.card_to_num(card2)

                    eval = Evaluation(Hand(card1, card2), [],
                                      GAME_TYPE.get_num_players())
                    result = eval.run_evaluation(
                        iterations=EVALUATION_ITERATIONS)

                    strengths.append(result.equity)
                    hands.append([num1, num2])

    strengths = np.array(strengths)
    stregths_idx_sorted = np.argsort(strengths)[::-1]

    hands = np.array(hands)
    sorted_hands = hands[stregths_idx_sorted]
    np.save(os.path.join(BASE_STARTING_HAND_DIR,
            STARTING_HAND_FILENAME), sorted_hands)


def calculate_hand_clashing_indexes():
    hands = np.load(os.path.join(
        BASE_STARTING_HAND_DIR, STARTING_HAND_FILENAME))
    clashes = np.zeros((0, 202))

    for hand in hands:
        indexes = []
        for j, other_hand in enumerate(hands):
            if hand[0] == other_hand[0] or hand[1] == other_hand[0] or hand[0] == other_hand[1] or hand[1] == other_hand[1]:
                indexes.append(j)
        clashes = np.append(clashes, np.array([indexes]), 0)
    np.save(os.path.join(BASE_STARTING_HAND_DIR,
            STARTING_HAND_CLASHES_FILENAME), clashes)


def print_top_x_hands(x):
    hands = np.load(os.path.join(
        BASE_STARTING_HAND_DIR, STARTING_HAND_FILENAME))

    for hand in hands[:x]:
        num1, num2 = hand[0], hand[1]

        card1, suit1 = math.ceil(num1 / 4), num1 % 4
        card2, suit2 = math.ceil(num2 / 4), num2 % 4
        if card1 == 13:
            card1 = 1
        else:
            card1 += 1

        if card2 == 13:
            card2 = 1
        else:
            card2 += 1

        card1 = Card(Suit.from_index(int(suit1 + 1)),
                     Value.from_index(int(card1)))
        card2 = Card(Suit.from_index(int(suit2 + 1)),
                     Value.from_index(int(card2)))

        print(Hand(card1, card2))


def get_position_hand_range(game_type: GameType, position, blinds):
    charts = load_range_charts(game_type, blinds)
    if position == Position.UTG2:
        charts = charts[Position.UTG1.value]
    else:
        charts = charts[position.value]

    hands = np.load(os.path.join(
        BASE_STARTING_HAND_DIR, STARTING_HAND_FILENAME))
    hands = [Hand(*sorted([Evaluation.num_to_card(card) for card in hand], reverse=True))
             for hand in hands]
    # hands = [Hand(Card(Suit.Spades, Value.Seven),
    #               Card(Suit.Hearts, Value.Five))]

    hand_probs = []
    for hand in hands:
        all_actions = []
        for action in charts.keys():
            for opponent in charts[action].keys():
                chart = charts[action][opponent]
                chart_actions = chart.get_actions()

                for a in chart_actions:
                    if a not in all_actions:
                        all_actions.append(a)

        probs = [0 for _ in range(len(all_actions))]
        for action in [OpponentAction.RFI, OpponentAction.RAISE]:
            if action.value not in charts:
                continue
            for opponent in charts[action.value].keys():
                chart = charts[action.value][opponent]
                chart_actions = chart.get_actions()
                for i, p in enumerate(chart.get_probs(hand)):
                    action_index = all_actions.index(chart_actions[i])
                    probs[action_index] += p

        total_p = sum(probs)
        fold_index = [all_actions.index(fold) for fold in ['Fold', 'FOLD']]
        fold_p = sum([probs[i] for i in fold_index])
        call_p = (total_p - fold_p)/total_p
        hand_probs.append(call_p)
        # print(f"{hand}: {call_p}")

    hand_probs = np.array(hand_probs)
    hand_probs /= hand_probs.sum()

    file_name = f"{game_type.get_num_players()}_{blinds}_{position.value}_{STARTING_HAND_PROBS}"
    path = os.path.join(BASE_STARTING_HAND_DIR, file_name)
    np.save(path, hand_probs)

    # for hand, prob in zip(hands, hand_probs):
    #     print(f"{hand}: {prob}")


def build_position_hand_ranges(game_type: GameType):
    blinds = [10, 30, 80]
    for blind in blinds:
        for position in GAME_TYPE_POSITIONS[game_type]:
            get_position_hand_range(game_type, position, blind)


if __name__ == "__main__":
    # calculate_hand_clashing_indexes()
    # print_top_x_hands(500)
    # get_position_hand_range(GameType.NinePlayer, Position.UTG1, 30)

    build_position_hand_ranges(GameType.NinePlayer)
