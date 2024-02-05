import os
import numpy as np
import torch
import torch.nn as nn

from data.img_transformers import TABLE_FINAL_DIMENTIONS
from networks import BASE_WIEGHT_DIR
from data.img_transformers import table_transformer
from poker.GameState import GameState


class StateDetector(nn.Module):
    def __init__(self, input_shape=TABLE_FINAL_DIMENTIONS, lr=0.001, player_count=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout2d(),
            nn.Flatten()
        )
        input = torch.zeros((1, *input_shape))
        output = self.encoder(input)
        output_dim = output.shape[1]

        self.player_card_fc = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 3)
        )

        self.table_card_fc = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 6)
        )

        self.dealer_pos_fc = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, player_count)
        )

        self.num_opponents_fc = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.Linear(64, player_count-1)
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        player_cards = self.player_card_fc(encoded)
        table_cards = self.table_card_fc(encoded)
        dealer_pos = self.dealer_pos_fc(encoded)
        opponents = self.num_opponents_fc(encoded)
        return player_cards, table_cards, dealer_pos, opponents

    def save(self, filename: str):
        filename = os.path.join(BASE_WIEGHT_DIR, f'{filename}.pth')
        torch.save(self.state_dict(), filename)

    @staticmethod
    def load(filename: str):
        filename = os.path.join(BASE_WIEGHT_DIR, f'{filename}.pth')
        model = StateDetector()
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict, assign=True)
        return model

    def get_state(self, screenshot) -> GameState:
        # to batch
        transformed = table_transformer(screenshot)
        batch = transformed.unsqueeze(0)
        batch = batch.to(torch.float32).to("cuda")

        player_preds, table_preds, dealer_pos, opponents = self.forward(
            batch)

        softmax = nn.Softmax(dim=1)
        sigmoid = nn.Sigmoid()

        player_count = torch.argmax(softmax(player_preds), 1)[0]
        table_count = torch.argmax(softmax(table_preds), 1)[0]
        dealer_pos = torch.argmax(softmax(dealer_pos), 1)[0]

        opponents = [(sigmoid(opponent) > 0.5).item()
                     for opponent in opponents[0]]

        return GameState(player_count.item(), table_count.item(), dealer_pos.item(), np.array(opponents))
