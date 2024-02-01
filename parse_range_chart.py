import os
import numpy as np
import json
from enums.GameType import GameType
from enums.Position import Position
from enums.Value import Value
from enums.Action import Action

from ranges import BASE_CHART_DIR
from ranges.RangeChart import load_range_charts

FILENAME = "test.json"
GAME_TYPE = GameType.SixPlayer
POSITION = Position.BB
ACTION = Action.RaiseSB

FOLD = "FOLD"


def get_all_actions(data):
    all_actions = []
    for hand in data:
        if 'actions' in data[hand]:
            actions = data[hand]["actions"].keys()
            for action in actions:
                if action not in all_actions:
                    all_actions.append(action)

    if FOLD not in all_actions:
        all_actions.append(FOLD)

    return all_actions


def parse_hand(hand):
    value_1 = Value.from_string(hand[0])
    value_2 = Value.from_string(hand[1])

    if len(hand) == 2:
        return value_1.value, value_2.value, 1

    return value_1.value, value_2.value, int(hand[2] == "s")


def get_action_values(hand, actions):
    hand_actions = np.zeros(len(actions))
    fold_idx = actions.index(FOLD)

    if "actions" not in hand:
        hand_actions[fold_idx] = 1.0
        return hand_actions

    for action, value in hand["actions"].items():
        action_idx = actions.index(action)
        hand_actions[action_idx] = float(value['val'])

    return hand_actions


def parse(json_str):
    data = json.loads(json_str)
    data = data["data_v2"]["data"]

    actions = get_all_actions(data)
    chart = np.zeros((15, 15, 2, len(actions)))

    for hand, value in data.items():
        idx1, idx2, idx3 = parse_hand(hand)
        hand_actions = get_action_values(value, actions)
        chart[idx1, idx2, idx3] = hand_actions

    return chart, actions


if __name__ == "__main__":
    # charts = load_range_charts()
    # chart = charts[GAME_TYPE][POSITION][ACTION]
    # hand = Hand(Card(Suit(3), Value(11)), Card(Suit(3), Value(10)))
    # print(chart[hand])
    with open(FILENAME, 'r') as f:
        chart, actions = parse(f.read())
        base_filename = f"{GAME_TYPE.value}-{POSITION.value}-{ACTION.value}"

        filename = f"{base_filename}-chart.npy"
        save_path = os.path.join(BASE_CHART_DIR, filename)
        np.save(save_path, chart)

        with open(os.path.join(BASE_CHART_DIR, f"{base_filename}-actions.txt"), 'w') as f:
            f.write(",".join(actions))
