import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from PIL import Image
import keras
from pypianoroll import Multitrack, Track
import pretty_midi
from keras.preprocessing.image import load_img, img_to_array
from azure.storage.blob import BlobServiceClient
import aubio
import librosa


def download_model_file(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open("model_save.keras", 'wb') as download_file:
        download_file.write(blob_client.download_blob().readall())

    return keras.models.load_model('model_save.keras')


def load_from_png(file_path, target_size=(72, 384)):
    img = load_img(file_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array.squeeze()


def load_from_midi(midi_file, fs=100, pitch_range=(0, 72)):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = midi_data.get_piano_roll(fs=fs)[pitch_range[0]:pitch_range[1], :]
    piano_roll = np.where(piano_roll > 0, 1, 0)
    return piano_roll


def load_from_pianoroll(file_path, track_index=0, binarize=True):
    multitrack = Multitrack(file_path)
    track = multitrack.tracks[track_index]
    piano_roll = track.pianoroll
    if binarize:
        piano_roll = np.where(piano_roll > 0, 1, 0)
    return piano_roll


def load_data_from_file(file_path):
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return load_from_png(file_path)
    elif file_path.lower().endswith(('.mid', '.midi')):
        return load_from_midi(file_path)
    elif file_path.lower().endswith(('.npz', '.npy')):
        return load_from_pianoroll(file_path)
    else:
        raise ValueError("Unsupported file type. The file must be an image (.png, .jpg, .jpeg), MIDI (.mid, .midi), or piano roll (.npz, .npy).")


def generate_variations(generator, base_noise, variations=10):
    outputs = []
    for i in range(variations):
        noise = base_noise + np.random.normal(0, 0.1, base_noise.shape)
        generated_image = generator.predict(noise[np.newaxis, :])
        generated_image = generated_image.squeeze(axis=0)
        outputs.append(generated_image)
    return outputs


def remove_short_notes(image, threshold=12):
    binarized = (image > 0.5).astype(np.float32)
    for pitch in range(binarized.shape[0]):
        active_note_start = None
        for time_step in range(binarized.shape[1]):
            if binarized[pitch, time_step] == 1 and active_note_start is None:
                # note start
                active_note_start = time_step
            elif binarized[pitch, time_step] == 0 and active_note_start is not None:
                # note end
                if time_step - active_note_start < threshold:
                    binarized[pitch, active_note_start:time_step] = 0
                active_note_start = None

        if active_note_start is not None and binarized.shape[1] - active_note_start < threshold:
            binarized[pitch, active_note_start:] = 0
    return binarized


# test blob loading:
azure_storage_connection_string="BlobEndpoint=https://vivygan.blob.core.windows.net/;QueueEndpoint=https://vivygan.queue.core.windows.net/;FileEndpoint=https://vivygan.file.core.windows.net/;TableEndpoint=https://vivygan.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-12-31T07:38:19Z&st=2024-04-17T22:38:19Z&spr=https&sig=tjF7obtcleBr%2BIyoPJ4dXKUQxwnyuXH6SOkaD6%2BSBVg%3D"
model_container_name="vivygan-model"
model_blob_name="variation_save.keras"

model = download_model_file(azure_storage_connection_string, model_container_name, model_blob_name)

# generator = keras.models.load_model('model/generator_save.keras')

file_path = 'maestro-v3.0.0_piano_rolls_pngs/2006/MIDI-Unprocessed_07_R1_2006_01-04_ORIG_MID--AUDIO_07_R1_2006_01_Track01_wav/MIDI-Unprocessed_07_R1_2006_01-04_ORIG_MID--AUDIO_07_R1_2006_01_Track01_wav_segment_87.png'
data = load_data_from_file(file_path)



base_noise = np.random.normal(0, 1, data.shape)

outputs = generate_variations(model, base_noise)
output_list = []
for output in outputs:
    binarized = remove_short_notes(output)
    output_list.append(binarized)

for i, img in enumerate(output_list):
    img = np.squeeze(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'L')
    img.save(f'generated/variations/variation_{i}.png')
