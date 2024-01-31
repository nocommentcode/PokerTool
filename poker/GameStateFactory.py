from data.img_transformers import poker_img_transformer
from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.Position import Position
from enums.TargetType import TargetType
from networks.CardDetector import CardDetector
from networks.PokerNetwork import PokerNetwork
from poker.State import FoldedGameState, PreFlopGameState


import pyautogui
import torch
from PIL.Image import Image


class GameStateFactory:
    def __init__(self, model: PokerNetwork, card_detector: CardDetector, game_type: GameType, pre_flop_charts, image_transformer=poker_img_transformer):
        self.model = model
        self.card_detector = card_detector
        self.transformer = image_transformer
        self.game_type = game_type
        self.pre_flop_charts = pre_flop_charts

    def get_game_state(self, screenshot, player_cards, table_cards):
        game_stage = self.get_game_stage(screenshot, player_cards, table_cards)

        if game_stage == GameStage.FOLDED:
            dealer_pos, * \
                _ = self.proccess_screenshot(
                    screenshot, get_player_cards=False, get_table_cards=False)
            return FoldedGameState(self.game_type, Position.from_dealer_pos(dealer_pos))

        elif game_stage == GameStage.PREFLOP:
            dealer_pos, player_cards, _ = self.proccess_screenshot(
                screenshot, get_player_cards=True, get_table_cards=False)

            position = Position.from_dealer_pos(dealer_pos)

            return PreFlopGameState(self.game_type, position, player_cards, self.pre_flop_charts[position])

    def get_game_stage(self, screenshot, player_cards, table_cards):
        if player_cards == 0:
            return GameStage.FOLDED

        if table_cards == 0:
            return GameStage.PREFLOP

        if table_cards == 3:
            return GameStage.FLOP

        if table_cards == 4:
            return GameStage.Turn

        if table_cards == 5:
            return GameStage.River

        return self.proccess_screenshot(screenshot)

    def proccess_screenshot(self, screenshot, get_player_cards=True, get_table_cards=True):
        transformed = self.transformer(screenshot)
        batch = transformed.unsqueeze(0)
        batch = batch.to(torch.float32).to("cuda")

        predictions = self.model.predict(batch)

        player_cards = None
        if get_player_cards:
            player_cards = [predictions[TargetType.Player_card_1.value].card,
                            predictions[TargetType.Player_card_2.value].card]

        table_cards = None

        dealer_pos = predictions[TargetType.Dealer_pos.value].value

        return dealer_pos, player_cards, table_cards

    def take_screenshot(self) -> Image:
        return pyautogui.screenshot()
