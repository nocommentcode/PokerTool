import numpy as np
import pandas as pd
from enums.GameType import GameType
from enums.Hand import Hand
from enums.OpponentAction import OpponentAction
from enums.Position import GAME_TYPE_POSITIONS, Position
from poker.BaseGameState import BaseGameState
from poker.GameState import GameState
from ranges.PreFlopEvaluation import PreFlopEvaluation


class PreFlopGameState(BaseGameState):
    def __init__(self, game_type: GameType, game_state: GameState, player_cards, charts):
        super().__init__(game_type, game_state)
        self.player_cards = player_cards
        self.hand = Hand(*player_cards)
        self.charts = charts

    def get_gto_ranges(self):
        columns = []
        rows = [pos.name for pos in GAME_TYPE_POSITIONS[self.game_type]]
        rows.remove(self.game_state.position.name)
        values = []

        rfi_action = None
        for action in sorted([OpponentAction(a) for a in self.charts.keys()]):
            if action == OpponentAction.RFI:
                rfi_action = self.charts[action.value]['none'][self.hand]
                continue

            columns.append(str(action))
            action_values = ["-" for _ in range(len(rows))]
            for opponent in self.charts[action.value].keys():
                index = rows.index(Position.from_string(opponent).name)
                action_values[index] = self.charts[action.value][opponent][self.hand]
            values.append(action_values)

        df = pd.DataFrame(np.array(values).swapaxes(0, 1), rows, columns)
        return df.head(len(rows)), rfi_action

    def str_gto(self):
        table, rfi = self.get_gto_ranges()
        return f"RFI: {rfi}\n{str(table)}"

    def str_player_cards(self):
        return f"Player Cards: {str(self.hand)}"

    def __str__(self) -> str:
        base = super().__str__()
        return "\n".join([base, self.str_player_cards(), "\n", self.str_gto()])
