from data.img_transformers import poker_img_transformer
from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.PokerTargetType import PLAYER_CARDS, TABLE_CARDS
from networks.PokerNetwork import PokerNetwork
from networks.StateDetector import StateDetector
from poker.FoldedGameState import FoldedGameState
from poker.PostFlopGameState import PostFlopGameState
from poker.PreFlopGameState import PreFlopGameState


import pyautogui
import torch


from time import sleep


class StateProvider:
    def __init__(self, state_detector: StateDetector, model: PokerNetwork, game_type: GameType, pre_flop_charts, image_transformer=poker_img_transformer):
        self.state_detector = state_detector
        self.model = model
        self.current_state = None
        self.transformer = image_transformer
        self.game_type = game_type
        self.pre_flop_charts = pre_flop_charts

    def take_screenshot(self):
        # return pyautogui.screenshot()

        from data import CLASSIFIED_DIR, UN_CLASSIFIED_DIR
        from PIL.Image import open as open_image
        import os
        return open_image(os.path.join(UN_CLASSIFIED_DIR, '1927f44a-6ea4-4791-8694-df8debc4c1ac.png'))

    def get_screenshot_and_state(self):
        screenshot = self.take_screenshot()
        state = self.state_detector.get_state(screenshot)
        return state, screenshot

    def get_next_state_consensus(self):
        state, _ = self.get_screenshot_and_state()
        sleep(0.05)
        validation_state, screenshot = self.get_screenshot_and_state()

        if state != validation_state:
            return self.get_next_state_consensus()

        return validation_state, screenshot

    def get_cards(self, screenshot, get_player_cards=True, get_table_cards=True):
        transformed = self.transformer(screenshot)
        batch = transformed.unsqueeze(0)
        batch = batch.to(torch.float32).to("cuda")

        predictions = self.model.predict(batch)

        player_cards = None
        if get_player_cards:
            player_cards = [
                predictions[card.value].card for card in PLAYER_CARDS]

        table_cards = None
        if get_table_cards:
            table_cards = [
                predictions[card.value].card for card in TABLE_CARDS]

        return player_cards, table_cards

    def get_game_state(self, screenshot):

        base_args = (self.game_type, self.current_state)

        if self.current_state.game_stage == GameStage.FOLDED:
            return FoldedGameState(*base_args)

        elif self.current_state.game_stage == GameStage.PREFLOP:
            player_cards, _ = self.get_cards(
                screenshot, get_player_cards=True, get_table_cards=False)

            return PreFlopGameState(*base_args, player_cards, self.pre_flop_charts[self.current_state.position])

        player_cards, table_cards = self.get_cards(
            screenshot, get_player_cards=True, get_table_cards=True)
        table_cards = table_cards[:self.current_state.table_card_count]
        return PostFlopGameState(*base_args, player_cards, table_cards)

    def tick(self):
        next_state, screenshot = self.get_next_state_consensus()
        if self.current_state != next_state:
            self.current_state = next_state

            game_state = self.get_game_state(screenshot)
            print(str(game_state))

    def print_for_screenshot_(self, screenshot):
        self.current_state = self.state_detector.get_state(screenshot)
        print(str(self.get_game_state(screenshot)))
