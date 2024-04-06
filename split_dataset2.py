import pretty_midi
import numpy as np
import os


def quantize_notes(midi_data, sixteenth_note_duration):
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            quantized_start = round(note.start / sixteenth_note_duration) * sixteenth_note_duration
            quantized_end = round(note.end / sixteenth_note_duration) * sixteenth_note_duration
            note.start, note.end = quantized_start, quantized_end


def adjust_segment_times(midi_data, start_time, end_time, sixteenth_note_duration):
    notes_start_times = [note.start for inst in midi_data.instruments for note in inst.notes if
                         start_time <= note.start < end_time]
    notes_end_times = [note.end for inst in midi_data.instruments for note in inst.notes if
                       start_time < note.end <= end_time]

    adjusted_start_time = start_time
    if notes_start_times:
        first_note_time = min(notes_start_times)
        if first_note_time - start_time > sixteenth_note_duration:
            adjusted_start_time = first_note_time - sixteenth_note_duration

    adjusted_end_time = adjusted_start_time + (end_time - start_time)

    return adjusted_start_time, adjusted_end_time


def create_segmented_midis(midi_data, bars_duration, sixteenth_note_duration, output_dir, base_filename):
    quantize_notes(midi_data, sixteenth_note_duration)

    total_length = midi_data.get_end_time()
    segment_starts = []
    segment_end = 0

    while segment_end < total_length:
        if not segment_starts:
            segment_start = 0
        else:
            segment_start = segment_starts[-1] + bars_duration

        adjusted_start, adjusted_end = adjust_segment_times(midi_data, segment_start, segment_start + bars_duration,
                                                            sixteenth_note_duration)
        segment_starts.append(adjusted_start)
        segment_end = adjusted_end

        if segment_end <= segment_start:
            break

    for i, start_time in enumerate(segment_starts[:-1]):
        if i + 1 < len(segment_starts) - 1:
            end_time = segment_starts[i + 1]
        else:
            continue

        segment_midi = pretty_midi.PrettyMIDI()
        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum,
                                                    name=instrument.name)
            for note in instrument.notes:
                if start_time <= note.start < end_time:
                    new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch,
                                                start=max(0, note.start - start_time),
                                                end=min(end_time - start_time, note.end - start_time))
                    new_instrument.notes.append(new_note)
            if new_instrument.notes:
                segment_midi.instruments.append(new_instrument)

        if segment_midi.instruments:
            segment_file_path = os.path.join(output_dir, f"{base_filename}_segment_{i}.mid")
            segment_midi.write(segment_file_path)


def process_midi_file(file_path, output_dir, bpm=120):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    sixteenth_note_duration = 60 / (bpm * 2)
    bars_duration = 4 * (60 / bpm) * 4

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    create_segmented_midis(midi_data, bars_duration, sixteenth_note_duration, output_dir, base_filename)


def process_all_midi_files(root_dir, output_root_dir, bpm=120):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                file_path = os.path.join(subdir, file)

                relative_path = os.path.relpath(subdir, root_dir)
                output_dir_base = os.path.join(output_root_dir, relative_path)
                file_base_name = os.path.splitext(file)[0]
                output_dir = os.path.join(output_dir_base, file_base_name)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                process_midi_file(file_path, output_dir, bpm=bpm)


if __name__ == "__main__":
    root_dir = 'maestro-v3.0.0'
    output_root_dir = 'maestro-v3.0.0_segments'
    process_all_midi_files(root_dir, output_root_dir)
