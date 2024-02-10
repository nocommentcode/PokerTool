import os
from pathlib import Path

import numpy as np
import pandas as pd
from data import DATASET_DIR
from data.PokerDataset import data_loader_factory
from enums.PokerTargetType import PokerTargetType
from enums.Suit import Suit
from enums.Value import Value

DATASET_NAME = "6_player_new_crop"

if __name__ == "__main__":
    dataset_dir = Path(os.path.join(DATASET_DIR, DATASET_NAME))
    train_loader, test_loader = data_loader_factory(
        dataset_dir, 0.99, batch_size=32)

    distribution = {type: np.zeros((len(Suit)+1, len(Value)+1), dtype=int)
                    for type in PokerTargetType}
    for _, target in train_loader:
        for type in PokerTargetType:
            suit, value, _ = target[type]
            for s, v in zip(suit, value):
                distribution[type][s, v] += 1

    for type, dist in distribution.items():
        dist[-1] = dist.sum(0)
        dist[:, -1] = dist.sum(1)
        df = pd.DataFrame(dist, [str(suit) for suit in Suit] + ["Total"], [
            str(value) for value in Value] + ["Total"])
        print(str(type))
        print(df.head(6))
        print("\n\n")
