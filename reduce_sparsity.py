import os
import numpy as np


def is_sparse(piano_roll, threshold=0.0625):
    total_steps = piano_roll.shape[0]
    active_steps = np.count_nonzero(piano_roll)
    if active_steps / total_steps < 1 - threshold:
        return True
    return False


def has_silence(piano_roll):
    return np.any(np.all(piano_roll == 0, axis=1))


def process_directory(root_dir):
    sparse_count = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(subdir, file)
                data = np.load(file_path)

                piano_roll = data['piano_roll']

                if has_silence(piano_roll):
                    print(f"Removing sparse file: {file_path}")
                    sparse_count += 1
                    print(f"Number of sparse files: {sparse_count}")
                    os.remove(file_path)


if __name__ == "__main__":
    dataset_root = "maestro-v3.0.0_piano_rolls"
    process_directory(dataset_root)
    print("Processing complete.")
