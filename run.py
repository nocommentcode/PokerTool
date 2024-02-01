import pyautogui
import torch
from enums.GameStage import GameStage
from enums.GameType import GameType
from enums.PokerTargetType import PLAYER_CARDS
from networks.StateDetector import StateDetector
from time import sleep

from networks.PokerNetwork import PokerNetwork
from poker.State import FoldedGameState, PreFlopGameState
from ranges.RangeChart import load_range_charts
from data.img_transformers import poker_img_transformer

MODEL_NAME = '6_player_pc_dp'
STATE_DETECTOR_NAME = "card_detector"
GAME_TYPE = GameType.SixPlayer


class StateProvider:
    def __init__(self, state_detector: StateDetector, model: PokerNetwork, game_type: GameType, pre_flop_charts, image_transformer=poker_img_transformer):
        self.state_detector = state_detector
        self.model = model
        self.current_state = None
        self.transformer = image_transformer
        self.game_type = game_type
        self.pre_flop_charts = pre_flop_charts

    def take_screenshot(self):
        return pyautogui.screenshot()
        # from data import CLASSIFIED_DIR
        # return open_image(os.path.join(UN_CLASSIFIED_DIR, '3e1597ea-8e32-4cef-9453-dd276c5039f5.png'))

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

        return player_cards, table_cards

    def get_game_state(self, screenshot):

        if self.current_state.game_stage == GameStage.FOLDED:
            return FoldedGameState(self.game_type, self.current_state.position)

        elif self.current_state.game_stage == GameStage.PREFLOP:
            player_cards, _ = self.get_cards(
                screenshot, get_player_cards=True, get_table_cards=False)

            return PreFlopGameState(self.game_type, self.current_state.position, player_cards, self.pre_flop_charts[self.current_state.position])

    def tick(self):
        next_state, screenshot = self.get_next_state_consensus()
        if self.current_state != next_state:
            self.current_state = next_state

            game_state = self.get_game_state(screenshot)
            print(str(game_state))


def run():
    state_detector = StateDetector.load(STATE_DETECTOR_NAME)
    state_detector.eval()

    model = PokerNetwork.load(MODEL_NAME, conv_channels=[
                              16, 32], fc_layers=[64])
    model.eval()

    charts = load_range_charts()
    charts = charts[GAME_TYPE]

    state_provider = StateProvider(state_detector, model, GAME_TYPE, charts)

    while True:
        state_provider.tick()
        sleep(1)


if __name__ == "__main__":
    with torch.no_grad():
        run()
