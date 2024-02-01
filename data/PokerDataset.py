from torch.utils.data.sampler import WeightedRandomSampler
from typing import List
from data.PokerTargetBatch import PokerTargetBatch
from data.PokerTarget import PokerTarget
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
import os
from data.StateTarget import StateTarget
from data.StateTargetBath import StateTargetBath
from enums.Value import Value


class PokerDataset(Dataset):
    def __init__(self, data_dir: Path, target_class=PokerTarget) -> None:
        self.data_dir = data_dir
        self.target_class = target_class
        self.samples = [f for f in data_dir.iterdir()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> (torch.Tensor, PokerTarget):
        sample = self.samples[index]

        image = self.get_sample_image(sample)
        target = self.get_sample_target(sample)

        return image, target

    def get_sample_image(self, sample_dir: Path) -> torch.Tensor:
        image_path = os.path.join(sample_dir, 'image.pt')
        image = torch.load(image_path)
        return image

    def get_sample_target(self, sample_path: Path) -> PokerTarget:
        uuid = str(sample_path).split("\\")[-1]
        classification_path = os.path.join(sample_path, 'classification.txt')
        with open(classification_path, 'r') as f:
            return self.target_class(f.readline(), uuid)


def poker_target_collate(batch):
    tensor_data = torch.zeros((len(batch), *batch[0][0].shape))
    for i, (item, _) in enumerate(batch):
        tensor_data[i] = item

    targets = [item[1] for item in batch]
    return tensor_data.to(torch.float32),  PokerTargetBatch(targets)


def state_target_collate(batch):
    tensor_data = torch.zeros((len(batch), *batch[0][0].shape))
    for i, (item, _) in enumerate(batch):
        tensor_data[i] = item

    targets = [item[1] for item in batch]
    return tensor_data.to(torch.float32),  StateTargetBath(targets)


def subset_with_non_empty_card(ds, targets_types):
    indicies = []
    for i in range(len(ds)):
        _, target = ds[i]

        all_not_empty = True
        for type in targets_types:
            value, _, _ = target[type]
            if value == 0:
                all_not_empty = False

        if all_not_empty:
            indicies.append(i)

    return Subset(ds, indicies)


def get_target_weights(ds, target_type):
    class_counts = torch.zeros(len(Value))
    class_belonging = torch.zeros(len(ds)).long()

    for i, (_, target) in enumerate(ds):
        _, value, _ = target[target_type]
        class_counts[value] += 1
        class_belonging[i] = value

    class_probs = torch.ones(len(ds))
    class_probs /= class_counts[class_belonging]

    return class_probs


def get_even_sampler(ds, target_types):
    weights = sum([get_target_weights(ds, target_type)
                  for target_type in target_types])
    return WeightedRandomSampler(weights=weights, num_samples=len(ds))


POKER_TARGET = "poker_target"
STATE_TARGET = "state_target"
TARGET_CLASSES = {
    POKER_TARGET: PokerTarget,
    STATE_TARGET: StateTarget
}

COLLAE_FUNC = {
    POKER_TARGET: poker_target_collate,
    STATE_TARGET: state_target_collate
}


def data_loader_factory(dataset_dir: Path, test_split: float, batch_size: int, target_type=POKER_TARGET, subsets: List[PokerTarget] = None):
    target_class = TARGET_CLASSES[target_type]
    collate_func = COLLAE_FUNC[target_type]

    ds = PokerDataset(dataset_dir, target_class)

    testing, training = random_split(ds, (1-test_split, test_split))

    train_sampler = None
    if subsets is not None:
        train_sampler = get_even_sampler(
            training, subsets)

    train_loader = DataLoader(training, batch_size, sampler=train_sampler,
                              collate_fn=collate_func, num_workers=4, persistent_workers=True)  # , prefetch_factor=5)
    test_loader = DataLoader(testing, batch_size,
                             collate_fn=collate_func, num_workers=4, persistent_workers=True)  # , prefetch_factor=5)

    return train_loader, test_loader
