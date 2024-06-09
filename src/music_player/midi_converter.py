from pathlib import Path

import numpy as np
from midiutil import MIDIFile

DATA_FOLDER = Path(__file__).parents[2] / "data"


# Function to create a MIDI file from pitch data
def create_midi_from_pitches(pitches, output_filename):
    num_tracks = pitches.shape[1]
    midi = MIDIFile(num_tracks)  # Four tracks for SATB voices
    tempo = 60 * 4  # Set a default tempo
    volume = 100  # Set a default volume

    # Add tempo to the MIDI file
    midi.addTempo(0, 0, tempo)

    # Initialize previous pitches and note start times
    prev_pitches = [-1] * num_tracks
    note_start_times = [0] * num_tracks

    # Add notes to the MIDI file
    for time_step, pitches_at_step in enumerate(pitches):
        for track, pitch in enumerate(pitches_at_step):
            if pitch == prev_pitches[track]:  # If the pitch is the same as the previous one
                continue  # Skip this iteration and increase the duration of the note
            else:  # If the pitch is different
                if prev_pitches[track] != -1:  # If there was a previous note
                    # Add the previous note with its duration
                    midi.addNote(track, 0, prev_pitches[track], note_start_times[track],
                                 time_step - note_start_times[track], volume)

                # Update the start time of the new note
                note_start_times[track] = time_step

            # Update the previous pitch
            prev_pitches[track] = pitch

    # Add the last notes
    for track, pitch in enumerate(prev_pitches):
        if pitch != -1:  # If there was a note
            # Add the note with its duration
            midi.addNote(track, 0, pitch, note_start_times[track], len(pitches) - note_start_times[track], volume)

    # Write the MIDI file to disk
    with open(DATA_FOLDER / output_filename, 'wb') as f:
        midi.writeFile(f)

    print(f"Audio saved to {output_filename}")


if __name__ == "__main__":
    num_tempi = 16
    num_tracks = 2

    song = np.random.randint(0, 128, (num_tempi, num_tracks))
    create_midi_from_pitches(song, "random.mid")
