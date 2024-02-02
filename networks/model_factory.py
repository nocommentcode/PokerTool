from enums.GameType import GameType
from networks.PokerNetwork import PokerNetwork
from networks.StateDetector import StateDetector


six_player_model_data = {
    "state_detector_name": "6_player_state_detector",
    "poker_network_name": "6_player_player_cards",
    "poker_network_conv": [32, 64],
    "poker_network_fc": [64]
}

game_mode_data = {
    GameType.SixPlayer: six_player_model_data
}


def model_factory(game_type: GameType) -> (StateDetector, PokerNetwork):
    data = game_mode_data[game_type]

    state_detector = StateDetector.load(data['state_detector_name'])
    state_detector.eval()

    model = PokerNetwork.load(
        data['poker_network_name'], conv_channels=data["poker_network_conv"], fc_layers=data["poker_network_fc"])
    model.eval()

    return state_detector, model
