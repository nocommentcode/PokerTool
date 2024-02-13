
import os
import re

import numpy as np
import pandas as pd
from enums.GameType import GameType
from enums.Hand import Hand
from enums.OpponentAction import OpponentAction
from enums.Position import GAME_TYPE_POSITIONS, Position
from ranges import BASE_CHART_DIR
from ranges.RangeChart import RangeChart
from utils.PrettyTable import PrettyTable
from utils.printing import red_text, green_text


class GTOPreflopRange:
    def __init__(self, game_type: GameType, blinds: int, position: Position):
        self.charts = {}
        self.rfi_chart = None
        self.position = position
        self.game_type = game_type
        self.load_charts(game_type, blinds, position)

    def load_chart_and_action(self, filename):
        chart = np.load(os.path.join(BASE_CHART_DIR, f"{filename}chart.npy"))

        with open(os.path.join(BASE_CHART_DIR, f"{filename}actions.txt"), 'r') as f:
            text = f.read()
            return chart,  text.split(",")

    def load_charts(self, game_type, blinds, position):
        chart_names = os.listdir(BASE_CHART_DIR)

        for chart_name in chart_names:
            values = re.search(
                fr"{game_type.value}-{blinds}-{position.value}-(?P<action>.*)-(?P<opponent>.*)-", chart_name)

            # not the right chart
            if values is None:
                continue

            action = values.group('action')
            opponent = values.group('opponent')

            filename = f"{game_type.value}-{blinds}-{position.value}-{action}-{opponent}-"
            chart, actions = self.load_chart_and_action(filename)
            range_chart = RangeChart(chart, actions)

            if action == OpponentAction.RFI.value:
                self.rfi_chart = range_chart
                continue

            if action not in self.charts:
                self.charts[action] = {}

            self.charts[action][opponent] = range_chart

    def __getitem__(self, hand):
        if type(hand) != Hand:
            raise ValueError("Hand should be hand object")

        opponent_positions = [
            pos.name for pos in GAME_TYPE_POSITIONS[self.game_type]]
        opponent_positions.remove(self.position.name)

        actions = []
        gto = []

        rfi_action = self.rfi_chart[hand] if self.rfi_chart is not None else '-'

        for action in sorted([OpponentAction(a) for a in self.charts.keys()]):
            actions.append(str(action))
            action_gto = ["-" for _ in range(len(opponent_positions))]

            for opponent in self.charts[action.value].keys():
                index = opponent_positions.index(
                    Position.from_string(opponent).name)
                action_gto[index] = self.charts[action.value][opponent][hand]
            gto.append(action_gto)

        def formatter(value):
            if value in ["FOLD", "Fold"]:
                return red_text(str(value))

            if value == "-":
                return "-"

            return green_text(str(value))

        table = PrettyTable("GTO Range", "blue", 3)
        table.add_row_names(opponent_positions)
        table.add_data(np.array(gto).swapaxes(0, 1), actions, formatter)

        return str(table), formatter(rfi_action)
