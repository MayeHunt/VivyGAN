import os
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_sampled_dataset_csv(root_directory, output_csv_path, sample_fraction=0.2, train_val_split=0.8,
                                 val_relative_split=0.125):
    file_paths = []
    set_labels = []

    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.npz'):
                file_paths.append(os.path.join(subdir, file))

    sampled_file_paths, _ = train_test_split(file_paths, test_size=1 - sample_fraction, random_state=42)

    train_val_paths, test_paths = train_test_split(sampled_file_paths, test_size=1 - train_val_split, random_state=42)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_relative_split, random_state=42)

    file_paths = train_paths + val_paths + test_paths
    set_labels = ['train'] * len(train_paths) + ['validation'] * len(val_paths) + ['test'] * len(test_paths)

    df = pd.DataFrame({'file_path': file_paths, 'set': set_labels})
    df.to_csv(output_csv_path, index=False)

    print(f"Sampled dataset CSV generated at {output_csv_path}. Sample size: {len(sampled_file_paths)} files.")


def generate_dataset_csv(root_directory, output_csv_path, train_val_split=0.8, val_relative_split=0.125):
    file_paths = []
    set_labels = []

    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.png'):
                file_paths.append(os.path.join(subdir, file))

    train_val_paths, test_paths = train_test_split(file_paths, test_size=1 - train_val_split, random_state=42)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_relative_split, random_state=42)

    file_paths = train_paths + val_paths + test_paths
    set_labels = ['train'] * len(train_paths) + ['validation'] * len(val_paths) + ['test'] * len(test_paths)

    df = pd.DataFrame({'file_path': file_paths, 'set': set_labels})
    df.to_csv(output_csv_path, index=False)

    print(f"Dataset CSV generated at {output_csv_path}.")


if __name__ == "__main__":
    # SAMPLE DATASET
    # root_directory = 'maestro-v3.0.0_piano_rolls'
    # output_csv_path = 'sampled_dataset_split.csv'
    # generate_sampled_dataset_csv(root_directory, output_csv_path)

    # FULL DATASET
    root_directory = 'maestro-v3.0.0_piano_rolls_pngs'
    output_csv_path = 'dataset_split.csv'
    generate_dataset_csv(root_directory, output_csv_path)

