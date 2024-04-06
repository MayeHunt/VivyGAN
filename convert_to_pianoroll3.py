import os
import pretty_midi
import numpy as np
from tqdm import tqdm


def midi_to_piano_roll(midi_path, fs=48, max_length=384):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    piano_roll = piano_roll[24:96]
    piano_roll = np.where(piano_roll > 0, 1, 0)
    if piano_roll.shape[1] > max_length:
        piano_roll = piano_roll[:, :max_length]
    elif piano_roll.shape[1] < max_length:
        padding_amount = max_length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, padding_amount)), 'constant', constant_values=0)
    return piano_roll


def process_directory(input_root, output_root):
    for subdir, dirs, files in os.walk(input_root):
        for file in tqdm([f for f in files if f.endswith(".mid") or f.endswith(".midi")], desc="Processing files"):
            input_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(subdir, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            piano_roll = midi_to_piano_roll(input_path).T

            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.npz')
            np.savez_compressed(output_path, piano_roll=piano_roll)


if __name__ == "__main__":
    input_dir = 'maestro-v3.0.0_segments'
    output_dir = 'maestro-v3.0.0_piano_rolls'

    process_directory(input_dir, output_dir)
