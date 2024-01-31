from os import listdir
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
from pathlib import Path
from PIL import Image
from PIL.Image import open
import matplotlib.pyplot as plt
from tqdm import tqdm

CROP_IMG_TOP_LEFT = (200, 230)
CROP_IMG_WIDTH = 1400
CROP_IMG_HEIGHT = 900

RESIZE_FACT = 0.2
FINAL_DIMENSIONS = (3, 180, 280)

IMG_MEANS = [0.1640, 0.2399, 0.1496]
IMG_STDS = [0.2273, 0.1977, 0.2046]


def custom_crop(image):
    x, y = CROP_IMG_TOP_LEFT
    image = image[:, y:y+CROP_IMG_HEIGHT, x:x+CROP_IMG_WIDTH]
    return image


to_tensor = transforms.PILToTensor()
crop = transforms.Lambda(custom_crop)
convert_to_float = transforms.ConvertImageDtype(dtype=torch.float32)
resize = transforms.Resize(
    (int(CROP_IMG_HEIGHT*RESIZE_FACT), int(CROP_IMG_WIDTH*RESIZE_FACT)), antialias=False)
normalize = transforms.Normalize(IMG_MEANS, IMG_STDS, inplace=True)
to_image = transforms.ToPILImage()


poker_img_transformer = transforms.Compose(
    [to_tensor, crop, resize, convert_to_float,  normalize])


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # load an image
    dir = Path("images/classified_images")
    remaining = listdir(dir)[1:]

    transformer = transforms.Compose(
        [to_tensor, crop, resize, convert_to_float, normalize])
    images = torch.zeros((len(remaining), *FINAL_DIMENSIONS))

    with tqdm(remaining) as all_remaining:
        for i, filename in enumerate(all_remaining):
            image = open(Path(f"{dir}/{filename}/image.png"))
            images[i] = transformer(image)

    print(images.mean((0, 2, 3)))
    print(images.std((0, 2, 3)))
