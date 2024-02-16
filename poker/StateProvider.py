import time
import uuid

import numpy as np
from data import UN_CLASSIFIED_DIR
from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.PokerTargetType import PLAYER_CARDS, TABLE_CARDS
from enums.Position import GAME_TYPE_POSITIONS
from enums.StackSize import StackSize
from networks.PokerNetwork import PokerNetwork
from networks.StateDetector import StateDetector
from poker.FoldedGameState import FoldedGameState
from poker.GameState import GameState
from poker.Player import Player
from poker.PostFlopGameState import PostFlopGameState
from poker.PreFlopGameState import PreFlopGameState
from data.img_transformers import table_transformer, cards_transformer
from PIL.Image import Image
import pyautogui
import torch


from time import sleep

from ranges.GTOPreflopRange import GTOPreflopRange


class StateProvider:
    def __init__(self, state_detector: StateDetector, model: PokerNetwork, game_type: GameType):
        self.state_detector = state_detector
        self.model = model
        self.current_state = None
        self.game_type = game_type
        self.blinds = 100
        self.load_gto_ranges(StackSize.Eighty)

    def load_gto_ranges(self, player_stack: StackSize):
        self.pre_flop_charts = {
            position: GTOPreflopRange(self.game_type, player_stack.value, position) for position in GAME_TYPE_POSITIONS[self.game_type]
        }

    def set_blinds(self, blinds):
        self.blinds = blinds
        self.current_state = None
        self.tick()

    def take_screenshot(self):
        # return pyautogui.screenshot()
        from PIL.Image import open as openimage
        return openimage("images/unclassified_images/df92cb62-d85b-479f-8b4e-30be9b5d64f3.png")

    def get_screenshot_and_state(self):
        screenshot = self.take_screenshot()
        player_count, table_count, dealer_pos, opponents = self.state_detector.predict(
            screenshot)
        return GameState(self.game_type, self.blinds, player_count, table_count, dealer_pos, opponents, screenshot)

    def get_next_state_consensus(self):
        state = self.get_screenshot_and_state()
        sleep(0.1)
        validation_state = self.get_screenshot_and_state()

        if state != validation_state:
            return self.get_next_state_consensus()

        state.parse_players(self.current_state)
        return state

    def get_cards(self, screenshot, get_player_cards=True, get_table_cards=True):
        transformed = cards_transformer(screenshot)
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

    def get_game_state(self, game_state: GameState):
        base_args = (self.game_type, game_state, self.blinds)

        if game_state.game_stage == GameStage.FOLDED:
            return FoldedGameState(*base_args)

        elif game_state.game_stage == GameStage.PREFLOP:
            player_cards, _ = self.get_cards(
                game_state.screenshot, get_player_cards=True, get_table_cards=False)

            return PreFlopGameState(*base_args, player_cards, self.pre_flop_charts[game_state.player.position])

        player_cards, table_cards = self.get_cards(
            game_state.screenshot, get_player_cards=True, get_table_cards=True)
        table_cards = table_cards[:game_state.table_card_count]
        return PostFlopGameState(*base_args, player_cards, table_cards)

    def tick(self, save_screenshots=False):
        next_state = self.get_next_state_consensus()

        if save_screenshots and self.should_save_screenshot(next_state):
            print(f"Saved screenshot")
            self.save_screenshot(next_state.screenshot)

        if self.current_state is None or self.current_state.player.stack_size != next_state.player.stack_size:
            self.load_gto_ranges(next_state.player.stack_size)

        if self.current_state != next_state:
            if self.should_refresh_ouput(next_state):
                game_state = self.get_game_state(next_state)
                print(str(game_state))

            self.current_state = next_state

    def should_save_screenshot(self, next_state: GameState):
        if self.current_state is None:
            return True

        if next_state.player_card_count != self.current_state.player_card_count:
            return True

        if next_state.table_card_count != self.current_state.table_card_count:
            return True

        return False

    def should_refresh_ouput(self, next_state: GameState):
        if self.current_state is None:
            return True

        if self.current_state.game_stage != next_state.game_stage:
            return True

        if self.current_state.game_stage == GameStage.FOLDED:
            return False

        if self.current_state.game_stage == GameStage.PREFLOP:
            return False

        return True

    def print_for_screenshot_(self, screenshot, game_stages=[gs for gs in GameStage]):
        start = time.time()
        player_count, table_count, dealer_pos, opponents = self.state_detector.predict(
            screenshot)
        state = GameState(self.game_type, self.blinds, player_count,
                          table_count, dealer_pos, opponents, screenshot)

        if state.game_stage in game_stages:
            state.parse_players(self.current_state)
            self.current_state = state

            print(str(self.get_game_state(state)))
            print(f"took: {str(time.time() - start)}")
            return True

        self.current_state = state
        return False

    def save_screenshot(self, screenshot):
        filename = f"{UN_CLASSIFIED_DIR}/{uuid.uuid4()}.png"
        screenshot.save(filename)
