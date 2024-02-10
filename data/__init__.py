from pathlib import Path

BASE_DATA_DIR = "images"

UN_CLASSIFIED_DIR = BASE_DATA_DIR + "/" + "unclassified_images"
Path(UN_CLASSIFIED_DIR).mkdir(parents=True, exist_ok=True)

CLASSIFIED_DIR = BASE_DATA_DIR + "/" + "classified_images"
Path(CLASSIFIED_DIR).mkdir(parents=True, exist_ok=True)

DATASET_DIR = BASE_DATA_DIR + "/" + "datasets"
Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)


IMAGE_NAME = "image.png"
POKER_TARGET_CLASSIFICATION = "poker_target.txt"
STATE_TARGET_CLASSIFICATION = "state_target.txt"
