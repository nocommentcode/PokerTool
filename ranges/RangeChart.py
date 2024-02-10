import os
from typing import Any, List
import numpy as np
import re
from enums.Action import Action
from enums.GameType import GameType

from enums.Hand import Hand
from enums.OpponentAction import OpponentAction
from enums.Position import Position

from ranges import BASE_CHART_DIR


class RangeChart:
    def __init__(self, chart: np.ndarray, actions: List[str]):
        self.chart = chart
        self.actions = actions

    def __getitem__(self, hand: Hand) -> Any:
        try:
            card_1, card_2 = hand.cards()

            ax_1 = card_1.value.value
            ax_2 = card_2.value.value
            suited_ax = int(hand.is_suited() or hand.is_pokets())

            probabilities = self.chart[ax_1, ax_2, suited_ax]
            action = np.random.choice(self.actions, p=probabilities)
            if action == "FOLD":
                return '-'
        except Exception as e:
            return f"{str(hand)} - Error ({probabilities})"

    def get_probs(self, hand: Hand):
        card_1, card_2 = hand.cards()

        ax_1 = card_1.value.value
        ax_2 = card_2.value.value
        suited_ax = int(hand.is_suited() or hand.is_pokets())

        return self.chart[ax_1, ax_2, suited_ax]

    def get_actions(self):
        return self.actions


def read_actions(game_type, blinds, position, action, opponent):
    with open(os.path.join(BASE_CHART_DIR, f"{game_type}-{blinds}-{position}-{action}-{opponent}-actions.txt"), 'r') as f:
        text = f.read()
        return text.split(",")


def load_range_charts(game_type, blinds):
    chart_names = os.listdir(BASE_CHART_DIR)
    charts = {}

    for chart_name in chart_names:
        values = re.search(
            fr"{game_type.value}-{blinds}-(?P<position>.*)-(?P<action>.*)-(?P<opponent>.*)-chart.npy", chart_name)

        # txt file
        if values is None:
            continue

        chart = np.load(os.path.join(BASE_CHART_DIR, chart_name))

        position = values.group('position')
        position = Position.from_string(position).value

        action = values.group('action')
        action = OpponentAction(action).value

        opponent = values.group('opponent')
        if opponent != 'none':
            opponent = Position.from_string(opponent).value

        actions = read_actions(game_type.value, blinds,
                               position, action, opponent)

        if position not in charts:
            charts[position] = {}

        if action not in charts[position]:
            charts[position][action] = {}

        charts[position][action][opponent] = RangeChart(
            chart, actions)

    return charts
