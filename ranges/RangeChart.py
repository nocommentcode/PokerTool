import os
from typing import Any, List
import numpy as np
import re
from enums.Action import Action
from enums.GameType import GameType

from enums.Hand import Hand
from enums.Position import Position

from ranges import BASE_CHART_DIR


class RangeChart:
    def __init__(self, chart: np.ndarray, actions: List[str]):
        self.chart = chart
        self.actions = actions

    def __getitem__(self, hand: Hand) -> Any:
        card_1, card_2 = hand.cards()

        ax_1 = card_1.value.value
        ax_2 = card_2.value.value
        suited_ax = int(hand.is_suited() or hand.is_pokets())

        probabilities = self.chart[ax_1, ax_2, suited_ax]
        return np.random.choice(self.actions, p=probabilities)


def read_actions(game_type, position, action):
    with open(os.path.join(BASE_CHART_DIR, f"{game_type.value}-{position.value}-{action.value}-actions.txt"), 'r') as f:
        text = f.read()
        return text.split(",")


def load_range_charts():
    chart_names = os.listdir(BASE_CHART_DIR)
    charts = {}

    for chart_name in chart_names:
        values = re.search(
            r"(?P<game_type>.*)-(?P<position>.*)-(?P<action>.*)-chart.npy", chart_name)

        # txt file
        if values is None:
            continue

        chart = np.load(os.path.join(BASE_CHART_DIR, chart_name))

        game_type = values.group('game_type')
        game_type = GameType(game_type)

        position = values.group('position')
        position = Position.from_string(position)

        action = values.group('action')
        action = Action(action)

        actions = read_actions(game_type, position, action)

        if game_type not in charts:
            charts[game_type] = {}

        if position not in charts[game_type]:
            charts[game_type][position] = {}

        charts[game_type][position][action] = RangeChart(chart, actions)

    return charts
