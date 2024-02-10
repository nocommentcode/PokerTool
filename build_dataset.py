import os
from pathlib import Path

from tqdm import tqdm
from data import CLASSIFIED_DIR, DATASET_DIR, IMAGE_NAME
from data.img_transformers import cards_transformer
import torch
from PIL.Image import open as open_image

DATASET_NAME = "6_player_new_crop"
DATA_DIR = CLASSIFIED_DIR


def build_dataset(dataset_name):
    classified_dir = Path(DATA_DIR)
    count = 0

    def get_image(sample):
        image_path = os.path.join(sample, IMAGE_NAME)
        image = open_image(image_path)
        return cards_transformer(image)

    def get_label(sample):
        classification_path = os.path.join(sample, "classification.txt")
        with open(classification_path, 'r') as f:
            return f.readline()

    with tqdm(os.listdir(classified_dir)) as dir:
        for uuid in dir:
            sample_dir = os.path.join(classified_dir, uuid)

            dataset_dir = Path(f"{DATASET_DIR}/{dataset_name}/{uuid}")
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # image
            image = get_image(sample_dir)
            torch.save(image, f"{dataset_dir}/image.pt")

            #  label
            label = get_label(sample_dir)
            with open(f"{dataset_dir}/classification.txt", 'w') as f:
                f.write(label)
                f.close()

            count += 1

    return count


if __name__ == "__main__":
    count = build_dataset(DATASET_NAME)
    print(f'Succesfully created {DATASET_NAME} with {count} samples')
