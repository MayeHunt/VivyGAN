import pandas as pd
import numpy as np


def load_data_from_csv(csv_path, set_type):
    df = pd.read_csv(csv_path)
    set_df = df[df['set'] == set_type]
    data = []

    total_files = len(set_df)
    print(f"Loading {total_files} {set_type} samples.")

    for idx, file_path in enumerate(set_df['file_path'], start=1):
        with np.load(file_path) as npz_file:
            piano_roll = npz_file['piano_roll']
            data.append(piano_roll)

        if idx % 100 == 0 or idx == total_files:
            print(f"Processed {idx}/{total_files} files.")

    return np.array(data)


if __name__ == "__main__":
    csv_path = 'dataset_split.csv'
    train_data = load_data_from_csv(csv_path, 'train')
    print(f"Loaded {len(train_data)} training samples.")
