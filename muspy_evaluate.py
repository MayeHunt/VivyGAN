import csv
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from keras.utils import img_to_array, load_img
import muspy
from azure.storage.blob import BlobServiceClient
import keras
from PIL import Image
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
    except RuntimeError as e:
        print(e)


def generate_noise(batch_size, latent_dim):
    return np.random.normal(0, 1, (batch_size, latent_dim))


def download_model_file(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open("model_save.keras", 'wb') as download_file:
        download_file.write(blob_client.download_blob().readall())

    return keras.models.load_model('model_save.keras')


def generate_random(generator, batch_size=10, latent_dim=100):
    noise = generate_noise(batch_size, latent_dim)
    generated_images = generator.predict(noise)

    binarized_images = (generated_images != 0).astype(int)
    print(binarized_images[0].shape)

    muspy_music_objects = []
    for i in range(batch_size):
        prepared_image = np.squeeze(binarized_images[i], axis=-1).T

        music = convert_to_muspy_class(prepared_image)
        muspy_music_objects.append(music)

    # metrics function not made yet, return this randomly picked one
    generated_track = muspy_music_objects[6]

    return generated_track


def load_generated_images(folder_path, threshold=127.0):
    images = []
    dir_size = os.listdir(folder_path)

    for i in range(len(dir_size)):
        filename = f"image_{i}.png"
        filepath = os.path.join(folder_path, filename)
        img = load_img(filepath, color_mode='grayscale')
        img = img_to_array(img)
        img = np.flipud(img)
        binarized_img = np.where(img > threshold, 1, 0).squeeze()
        images.append(binarized_img)

    return images


def convert_to_muspy_class(track_array, time_unit=1, pitch_offset=24):
    music = muspy.Music()
    track = muspy.Track(program=0, is_drum=False)
    for time, pitch_row in enumerate(track_array):
        for pitch, active in enumerate(pitch_row):
            if (active == 1).any():
                note = muspy.Note(
                    pitch=pitch + pitch_offset,
                    time=time * time_unit,
                    duration=time_unit,
                    velocity=100
                )
                track.notes.append(note)
    music.tracks.append(track)
    return music


def collect_metrics_and_write_to_csv(generated_songs, csv_filename):
    generated_songs_data = []
    for i, song in enumerate(generated_songs):
        song_data = {
            "Song": f"image_{i}.png",
            "Polyphony": muspy.polyphony(song),
            "Polyphony Rate": muspy.polyphony_rate(song),
            "Pitch in Scale Rate": muspy.pitch_in_scale_rate(song, root=46, mode='major'),
            "Scale Consistency": muspy.scale_consistency(song),
            "No. Pitches Used": muspy.n_pitches_used(song),
            "No. Pitch Classes Used": muspy.n_pitch_classes_used(song),
            "Pitch Class Entropy": muspy.pitch_class_entropy(song),
            "Pitch Entropy": muspy.pitch_entropy(song),
        }
        generated_songs_data.append(song_data)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=generated_songs_data[0].keys())
        writer.writeheader()
        for song_data in generated_songs_data:
            writer.writerow(song_data)
    print(f"Data written to {csv_filename}.")


if __name__ == '__main__':
    # test blob loading:
    azure_storage_connection_string = "BlobEndpoint=https://vivygan.blob.core.windows.net/;QueueEndpoint=https://vivygan.queue.core.windows.net/;FileEndpoint=https://vivygan.file.core.windows.net/;TableEndpoint=https://vivygan.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-12-31T07:38:19Z&st=2024-04-17T22:38:19Z&spr=https&sig=tjF7obtcleBr%2BIyoPJ4dXKUQxwnyuXH6SOkaD6%2BSBVg%3D"
    model_container_name = "vivygan-model"
    model_blob_name = "generator_save.keras"

    model = download_model_file(azure_storage_connection_string, model_container_name, model_blob_name)

    generated_songs = generate_random(generator=model)

    # images = load_generated_images('evaluation/fake')
    # generated_songs = [convert_to_muspy_class(image) for image in images]

    # muspy.show_pianoroll(generated_songs[10])
    # plt.show()

    # csv_filename = "generated_songs_metrics.csv"
    # collect_metrics_and_write_to_csv(generated_songs, csv_filename)

    # muspy.download_musescore_soundfont()
    muspy.write_audio('test.wav', generated_songs, audio_format='wav')
