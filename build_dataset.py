import os
from pathlib import Path

from tqdm import tqdm
from data import DATASET_DIR
from data.PokerTarget import PokerTarget
from data.img_transformers import poker_img_transformer
import torch
from PIL.Image import open as open_image

from enums.TargetType import TargetType


def build_dataset(dataset_name):
    classified_dir = Path("images/classified_images")
    count = 0

    def get_image(sample):
        image_path = os.path.join(sample, 'image.png')
        image = open_image(image_path)
        return poker_img_transformer(image)

    def get_label(sample):
        classification_path = os.path.join(sample, 'classification.txt')
        with open(classification_path, 'r') as f:
            return f.readline()

    with tqdm(os.listdir(classified_dir)) as dir:
        for uuid in dir:
            sample_dir = os.path.join(classified_dir, uuid)
            image = get_image(sample_dir)
            label = get_label(sample_dir)

            dataset_dir = Path(f"{DATASET_DIR}/{dataset_name}/{uuid}")
            dataset_dir.mkdir(parents=True, exist_ok=True)

            torch.save(image, f"{dataset_dir}/image.pt")

            with open(f"{dataset_dir}/classification.txt", 'w') as f:
                f.write(label)
                f.close()

            count += 1

    return count


if __name__ == "__main__":
    name = "6_player"
    count = build_dataset(name)
    print(f'Succesfully created {name} with {count} samples')
