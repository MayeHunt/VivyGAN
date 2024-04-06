import os
import numpy as np
from pypianoroll import Multitrack
import matplotlib.pyplot as plt
from PIL import Image

dataset_root = "maestro-v3.0.0_piano_rolls"
dest_root = "maestro-v3.0.0_piano_rolls_pngs"


def piano_roll_to_png(piano_roll, dest_file_path):
    piano_roll = (piano_roll > 0).astype(np.uint8) * 255
    image = Image.fromarray(piano_roll, 'L')  # 'L' mode for grayscale
    image.save(dest_file_path)


def process_directory(directory_path, destination_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                piano_roll = data['piano_roll']

                dest_file_path = file_path.replace(dataset_root, dest_root)
                dest_file_path = dest_file_path.replace(".npz", ".png")
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                piano_roll_to_png(piano_roll.T, dest_file_path)


if __name__ == "__main__":
    process_directory(dataset_root, dest_root)
