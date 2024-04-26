import os
import pandas as pd
import shutil
from random import sample


if __name__ == "__main__":
    csv_file_path = 'dataset_split.csv'
    target_dir = os.path.expanduser('evaluation/real')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    df = pd.read_csv(csv_file_path)
    sampled_df = df.sample(n=100, random_state=42)

    for _, row in sampled_df.iterrows():
        source_path = row['file_path']
        destination_path = os.path.join(target_dir, os.path.basename(source_path))
        shutil.copy2(source_path, destination_path)

    print(f'Copied {len(sampled_df)} images to {target_dir}')
