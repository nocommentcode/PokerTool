from pathlib import Path

BASE_CHART_DIR = "charts"
Path(BASE_CHART_DIR).mkdir(parents=True, exist_ok=True)

BASE_STARTING_HAND_DIR = "starting_hands"
Path(BASE_STARTING_HAND_DIR).mkdir(parents=True, exist_ok=True)

STARTING_HAND_FILENAME = "starting_hands.npy"
STARTING_HAND_CLASHES_FILENAME = "starting_hands_clashes.npy"
