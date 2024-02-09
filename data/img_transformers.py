from os import listdir
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
from PIL.Image import open
import matplotlib.pyplot as plt
from tqdm import tqdm

# shared
to_tensor = transforms.PILToTensor()
to_image = transforms.ToPILImage()
convert_to_float = transforms.ConvertImageDtype(dtype=torch.float32)

# Table
WINDOW_TOP_LEFT = (8, 15)
WINDOW_SIZE = (1775, 1267)


def crop_table(image):
    x_off, y_off = WINDOW_TOP_LEFT
    x_delta, y_delta = WINDOW_SIZE
    return image[:, y_off:y_off+y_delta, x_off:x_off+x_delta]


crop_table_transform = transforms.Lambda(crop_table)

TABLE_RESIZE_FACT = 0.1
table_resize = transforms.Resize(
    (int(WINDOW_SIZE[1]*TABLE_RESIZE_FACT), int(WINDOW_SIZE[0]*TABLE_RESIZE_FACT)), antialias=True)

TABLE_MEANS = [0.1605, 0.1998, 0.1435]
TABLE_STDS = [0.1771, 0.1554, 0.1461]
table_normalise = transforms.Normalize(TABLE_MEANS, TABLE_STDS, inplace=True)

TABLE_FINAL_DIMENTIONS = (3, 126, 177)
table_transformer = transforms.Compose(
    (to_tensor, crop_table_transform, table_resize, convert_to_float, table_normalise)
)


# Cards

CARDS_TOP_LEFT = (522, 509)
CARDS_SIZE = (728, 530)


def crop_cards(image):
    x_off, y_off = CARDS_TOP_LEFT
    x_delta, y_delta = CARDS_SIZE
    return image[:, y_off:y_off+y_delta, x_off:x_off+x_delta]


crop_cards_transform = transforms.Lambda(crop_cards)

CARDS_RESIZE_FACT = 0.2
cards_resize = transforms.Resize(
    (int(CARDS_SIZE[1]*CARDS_RESIZE_FACT), int(CARDS_SIZE[0]*CARDS_RESIZE_FACT)), antialias=True)

CARD_MEANS = [0.2056, 0.3482, 0.1962]
CARD_STDS = [0.2654, 0.2083, 0.2475]
cards_normalize = transforms.Normalize(CARD_MEANS, CARD_STDS, inplace=True)

CARDS_FINAL_DIMENTIONS = (3, 106, 145)
cards_transformer = transforms.Compose(
    (to_tensor, crop_table_transform,
     crop_cards_transform, cards_resize, convert_to_float, cards_normalize)
)

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cards_transformer = transforms.Compose(
        (to_tensor,  crop_table_transform,
         crop_cards_transform, cards_resize, convert_to_float, to_image))
    test = transforms.Compose(
        (to_tensor, transforms.Resize((1600, 2560), antialias=False), to_image))
    img = open(os.path.join("images/unclassified_images",
                            "25d0b29a-6ad3-48a0-a45a-adb01b761b8a.png"))
    # "678edd6f-eec3-4468-855c-b34c9838728f.png"))

    # img = test(img)
    plt.imshow(img)
    plt.show()

    img = cards_transformer(img)
    plt.imshow(img)
    plt.show()
    dfsd
    # load an image
    dir = Path("images/classified_images")
    remaining = listdir(dir)

    images = torch.zeros((len(remaining), *CARDS_FINAL_DIMENTIONS))
    with tqdm(remaining) as all_remaining:
        for i, filename in enumerate(all_remaining):
            image = open(Path(f"{dir}/{filename}/image.png"))
            image = cards_transformer(image)
            images[i] = image

    print(images.mean((0, 2, 3)))
    print(images.std((0, 2, 3)))
