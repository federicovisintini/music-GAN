from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.config import TRACKS, TEMPI, VOC_SIZE
from src.gan.discriminator import Discriminator
from src.gan.generator import ConvolutionalGenerator
from src.music_player.midi_converter import create_midi_from_pitches
from src.prepare_data.load_data import get_songs
from src.prepare_data.utils import encode, decode
from src.visualize.visualize_data import visualize_song

DATA_FOLDER = Path(__file__).parents[1] / "data"

if __name__ == "__main__":
    NUM_SONGS = 4

    songs = get_songs(tracks=TRACKS, tempi=TEMPI)
    songs = songs[:NUM_SONGS]
    songs = encode(songs)

    # Initialize generator and discriminator
    generator = ConvolutionalGenerator(vocab_size=VOC_SIZE, seq_length=TEMPI, voices=TRACKS)
    discriminator = Discriminator(vocab_size=VOC_SIZE, seq_length=TEMPI, voices=TRACKS)

    # load the weights
    generator.load_state_dict(torch.load(DATA_FOLDER / "model_generator.pth"))
    discriminator.load_state_dict(torch.load(DATA_FOLDER / "model_discriminator.pth"))

    # generate songs
    fake_songs = generator.generate(batch_size=NUM_SONGS)
    print("Generated songs", fake_songs.shape)

    # save the songs to midi
    for i, fake_song in enumerate(fake_songs):
        create_midi_from_pitches(decode(fake_song), f"generated_song_{i}.mid")

    # discriminate fake song
    prob = discriminator(fake_songs)
    print("Fake Songs are real with probability", prob.flatten().detach().numpy())

    # discriminate real song
    prob = discriminator(songs)
    print("Real Songs are real with probability", prob.flatten().detach().numpy())

    # print(fake_songs[0])
    # print(songs[0])

    visualize_song(songs[0], title="Real Song")
    visualize_song(fake_songs[0], title="Fake Song - Trained")
    plt.show()
