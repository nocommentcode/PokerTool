
from PIL.Image import open as open_image
from tqdm import tqdm
from pathlib import Path
import os
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def build_dataset():
    classified_dir = Path("images/classified_images")

    def get_image(sample):
        image_path = os.path.join(sample, 'image.png')
        image = open_image(image_path)
        return image.crop((748, 470, 1069, 518))

    with tqdm(os.listdir(classified_dir)) as dir:
        for uuid in dir:
            sample_dir = os.path.join(classified_dir, uuid)
            image = get_image(sample_dir)
            print(pytesseract.image_to_string(image))
            plt.imshow(image)
            plt.show()


if __name__ == "__main__":
    count = build_dataset()
