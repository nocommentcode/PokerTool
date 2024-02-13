import uuid
from data import UN_CLASSIFIED_DIR
from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.PokerTargetType import PLAYER_CARDS, TABLE_CARDS
from enums.Position import GAME_TYPE_POSITIONS
from networks.PokerNetwork import PokerNetwork
from networks.StateDetector import StateDetector
from poker.FoldedGameState import FoldedGameState
from poker.GameState import GameState
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
        self.blinds = 80
        self.load_gto_ranges()

    def load_gto_ranges(self):
        self.pre_flop_charts = {
            position: GTOPreflopRange(self.game_type, self.blinds, position) for position in GAME_TYPE_POSITIONS[self.game_type]
        }

    def set_blinds(self, blinds):
        self.blinds = blinds
        self.load_gto_ranges()
        self.current_state = None
        self.tick()

    def take_screenshot(self):
        # return pyautogui.screenshot()

        from data import CLASSIFIED_DIR, UN_CLASSIFIED_DIR
        from PIL.Image import open as open_image
        import os
        return open_image(os.path.join(CLASSIFIED_DIR, "3d49c957-4769-403b-aa38-338ce354818b", 'image.png'))

    def get_screenshot_and_state(self):
        screenshot = self.take_screenshot()
        state = self.state_detector.get_state(screenshot)
        return state, screenshot

    def get_next_state_consensus(self):
        state, _ = self.get_screenshot_and_state()
        sleep(0.1)
        validation_state, screenshot = self.get_screenshot_and_state()

        if state != validation_state:
            return self.get_next_state_consensus()

        return validation_state, screenshot

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

    def get_game_state(self, game_state: GameState, screenshot: Image):

        base_args = (self.game_type, game_state, self.blinds)

        if game_state.game_stage == GameStage.FOLDED:
            return FoldedGameState(*base_args)

        elif game_state.game_stage == GameStage.PREFLOP:
            player_cards, _ = self.get_cards(
                screenshot, get_player_cards=True, get_table_cards=False)

            return PreFlopGameState(*base_args, player_cards, self.pre_flop_charts[game_state.position])

        player_cards, table_cards = self.get_cards(
            screenshot, get_player_cards=True, get_table_cards=True)
        table_cards = table_cards[:game_state.table_card_count]
        return PostFlopGameState(*base_args, player_cards, table_cards)

    def tick(self, save_screenshots=False):
        next_state, screenshot = self.get_next_state_consensus()

        if save_screenshots and self.should_save_screenshot(next_state):
            print(f"Saved screenshot")
            self.save_screenshot(screenshot)

        if self.current_state != next_state:
            if self.should_refresh_ouput(next_state):
                game_state = self.get_game_state(next_state, screenshot)
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

    def print_for_screenshot_(self, screenshot):
        self.current_state = self.state_detector.get_state(screenshot)
        print(str(self.get_game_state(screenshot)))

    def save_screenshot(self, screenshot):
        filename = f"{UN_CLASSIFIED_DIR}/{uuid.uuid4()}.png"
        screenshot.save(filename)
