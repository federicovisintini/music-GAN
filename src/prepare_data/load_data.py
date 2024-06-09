from pathlib import Path

import numpy as np

DATA_FOLDER = Path(__file__).parents[2] / "data"


def extract_pieces(song: np.array, tempi: int = None):
    if tempi is None:
        return [song]

    # Calculate the number of arrays needed
    num_pieces = np.ceil(song.shape[0] / tempi).astype(int)

    # Initialize the list of arrays
    arrays = []

    for i in range(num_pieces):
        # Calculate the start and end indices for slicing
        start = i * tempi
        end = start + tempi

        # If the end index is beyond the last element of the array, pad the array
        if end > song.shape[0]:
            pad_width = ((0, end - song.shape[0]), (0, 0))
            array = np.pad(song[start:], pad_width=pad_width, mode='constant', constant_values=-1)
        else:
            array = song[start:end]

        arrays.append(array)

    return arrays


def select_tracks(song, tracks):
    assert tracks is None or 1 <= tracks <= 4, "Number of tracks must be between 1 and 4."

    if tracks == 1:
        return song[:, :1]
    elif tracks == 2:
        return np.stack((song[:, 0], song[:, -1])).T
    elif tracks == 3:
        return np.stack((song[:, 0], song[:, 1], song[:, -1])).T
    else:
        return song


def get_songs(tracks=None, tempi=None):
    jsf = np.load(DATA_FOLDER / 'music.npz', allow_pickle=True, encoding='latin1')
    songs = [select_tracks(song=song.astype(int), tracks=tracks) for song in jsf['pitches']]
    return [piece for song in songs for piece in extract_pieces(song, tempi=tempi)]


if __name__ == "__main__":
    song = get_songs(tracks=1, tempi=16)
    print(song[0])
