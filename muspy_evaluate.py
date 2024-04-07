import os
import numpy as np
from keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt

import muspy

def load_generated_images(folder_path, threshold=127.0):
    images = []
    files = os.listdir(folder_path)
    sorted_files = sorted(files)
    for filename in sorted_files:
        if filename.endswith('.png'):
            filepath = os.path.join(folder_path, filename)
            img = load_img(filepath, color_mode='grayscale')
            img = img_to_array(img)
            img = np.flipud(img)
            binarized_img = np.where(img > threshold, 1, 0).squeeze()
            images.append(binarized_img)
    return images


def convert_to_muspy_class(track_array, time_unit=1, pitch_offset=24):
    music = muspy.Music()
    track_array = track_array.T
    track = muspy.Track(program=0, is_drum=False)
    for time, pitch_row in enumerate(track_array):
        for pitch, active in enumerate(pitch_row):
            if active == 1:
                note = muspy.Note(
                    pitch=pitch + pitch_offset,
                    time=time * time_unit,
                    duration=time_unit,
                    velocity=100
                )
                track.notes.append(note)
    music.tracks.append(track)
    return music


if __name__ == '__main__':
    images = load_generated_images('evaluation/fake')
    generated_songs = [convert_to_muspy_class(image) for image in images]

    muspy.show_pianoroll(generated_songs[25])
    plt.show()

    print(muspy.polyphony(generated_songs[25]))
    print(muspy.pitch_in_scale_rate(generated_songs[25], root=46, mode='major'))
    print(muspy.scale_consistency(generated_songs[25]))
    muspy.download_musescore_soundfont()
    muspy.write_audio('test.wav', generated_songs[25], audio_format='wav')
