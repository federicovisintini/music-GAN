from src.music_player.midi_converter import create_midi_from_pitches
from src.prepare_data.load_data import get_songs


def save_song_to_file(num, tracks=None):
    pitches = get_songs(tracks=tracks)
    create_midi_from_pitches(pitches[num], f"song_{num}_{tracks}.mid")


if __name__ == "__main__":
    save_song_to_file(0, tracks=2)
