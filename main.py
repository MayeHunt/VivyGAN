import pypianoroll
from music21 import stream, note, midi, environment
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pypianoroll import Track, Multitrack

env = environment.Environment()


# This was mostly just a test of writing and playing midi


def write_midi():
    melody = stream.Stream()

    notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
    for pitch in notes:
        melody.append(note.Note(pitch))

    melody.write('midi', fp='melody.mid')


def play_midi():
    midi_file = 'test.mid'

    mf = midi.MidiFile()
    mf.open(midi_file)
    mf.read()
    mf.close()

    s = midi.translate.midiFileToStream(mf)

    s.show('midi')


def load_sparse_npz():
    npz_data = np.load(
        'maestro-v3.0.0_piano_rolls/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_segment_0.npz')
    print(list(npz_data.keys()))

    sparse_piano_roll_path = 'maestro-v3.0.0_piano_rolls/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_segment_0.npz'
    sparse_piano_roll = sp.load_npz(sparse_piano_roll_path)
    dense_piano_roll = sparse_piano_roll.todense().T

    track = pypianoroll.Track(pianoroll=dense_piano_roll, program=0, is_drum=False, name='Piano')
    multitrack = pypianoroll.Multitrack(tracks=[track])

    pypianoroll.plot_multitrack(multitrack, None)
    plt.show()


if __name__ == '__main__':
    npz_path = 'maestro-v3.0.0_piano_rolls/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_segment_3.npz'  # Update this path
    data = np.load(npz_path)
    piano_roll = data['piano_roll']
    print(piano_roll.shape)

    plt.imshow(piano_roll.T, aspect='auto', origin='lower')
    plt.xlabel('Time Steps')
    plt.ylabel('MIDI Note Number')
    plt.show()

    track = Track(pianoroll=piano_roll, program=0, is_drum=False, name="NPZ Piano Roll")
    multitrack2 = Multitrack(tracks=[track])

    pypianoroll.plot_multitrack(multitrack2, None)

    multitrack = pypianoroll.read('maestro-v3.0.0_segments/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_segment_3.mid')
    print(multitrack[0])
    pypianoroll.plot_multitrack(multitrack, None)
    plt.show()
