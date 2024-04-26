import numpy as np
from PIL import Image
from pretty_midi import PrettyMIDI, Instrument, Note
from midi2audio import FluidSynth
import winsound


def load_npz_and_play(npz_path):
    data = np.load(npz_path)
    piano_roll = data['piano_roll']
    midi = PrettyMIDI()
    instrument = Instrument(program=0)

    for time, note in np.ndenumerate(piano_roll):
        if note:
            midi_note = Note(velocity=100, pitch=time[1], start=time[0] * 0.5, end=(time[0] + 1) * 0.5)
            instrument.notes.append(midi_note)

    midi.instruments.append(instrument)

    midi_file = "output.mid"
    audio_file = "output.wav"
    midi.write(midi_file)
    fs = FluidSynth()
    fs.midi_to_audio(midi_file, audio_file)
    play_wav(audio_file)


def play_wav(file_path):
    winsound.PlaySound(file_path, winsound.SND_FILENAME)


def png_to_piano_roll_and_play(png_path):
    img = Image.open(png_path).convert('L')
    img_array = np.array(img)

    midi = PrettyMIDI()
    instrument = Instrument(program=0)

    threshold = 128
    for x in range(img_array.shape[1]):
        for y in range(img_array.shape[0]):
            if img_array[y, x] < threshold:
                note_pitch = 127 - y
                midi_note = Note(
                    velocity=100,
                    pitch=note_pitch,
                    start=x * 0.1,
                    end=(x + 1) * 0.1
                )
                instrument.notes.append(midi_note)

    midi.instruments.append(instrument)

    midi_file = "output_image.mid"
    audio_file = "output_image.wav"
    midi.write(midi_file)

    fs = FluidSynth()
    fs.midi_to_audio(midi_file, audio_file)

    play_wav(audio_file)


def play_wav(file_path):
    winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)


if __name__ == "__main__":
    png_path = "path.png"
    png_to_piano_roll_and_play(png_path)