import os
import torch
import torch.nn as nn
import torch.nn.functional as f

from data.img_transformers import FINAL_DIMENSIONS
from enums.TargetType import TargetType
from networks import BASE_WIEGHT_DIR
from data.img_transformers import poker_img_transformer


class CardDetector(nn.Module):
    def __init__(self, input_shape=FINAL_DIMENSIONS, lr=0.001):
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

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        player_cards = self.player_card_fc(encoded)
        table_cards = self.table_card_fc(encoded)
        return player_cards, table_cards

    def save(self, filename: str):
        filename = os.path.join(BASE_WIEGHT_DIR, f'{filename}.pth')
        torch.save(self.state_dict(), filename)

    @staticmethod
    def load(filename: str):
        filename = os.path.join(BASE_WIEGHT_DIR, f'{filename}.pth')
        model = CardDetector()
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict, assign=True)
        return model

    def get_card_counts(self, screenshot):
        transformed = poker_img_transformer(screenshot)
        batch = transformed.unsqueeze(0)
        batch = batch.to(torch.float32).to("cuda")
        player_preds, table_preds = self.forward(batch)

        softmax = nn.Softmax(dim=1)
        player_count = torch.argmax(softmax(player_preds), 1)[0]
        table_count = torch.argmax(softmax(table_preds), 1)[0]

        return player_count.item(), table_count.item()
