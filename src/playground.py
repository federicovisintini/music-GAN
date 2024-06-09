import time

from matplotlib import pyplot as plt

from src.config import VOC_SIZE, TEMPI, TRACKS
from src.gan.discriminator import Discriminator
from src.gan.generator import ConvolutionalGenerator
from src.music_player.midi_converter import create_midi_from_pitches
from src.prepare_data.load_data import get_songs
from src.prepare_data.utils import decode, encode
from src.visualize.visualize_data import visualize_song

# GENERATOR
generator = ConvolutionalGenerator(vocab_size=VOC_SIZE, seq_length=TEMPI, voices=TRACKS)
start = time.time()
fake_music = generator.generate(batch_size=1)  # output size [BATCH_SIZE, TEMPI, TRACKS]
print("Time to generate:", time.time() - start)
create_midi_from_pitches(decode(fake_music[0]), "random_0.mid")
print("Number of generator parameters:", sum(p.numel() for p in generator.parameters()))
print("Fake music shape", fake_music[0].shape)
print()

# DISCRIMINATOR
discriminator = Discriminator(vocab_size=VOC_SIZE, seq_length=TEMPI, voices=TRACKS)
start = time.time()
probas = discriminator(fake_music)
print("Time to discriminate:", time.time() - start)
print("Number of discriminator parameters:", sum(p.numel() for p in discriminator.parameters()))
print("Proba fake song is real", probas.item())
# print([p.numel() for p in discriminator.parameters()])

# DISCRIMINATOR - real song
songs = get_songs(tracks=TRACKS, tempi=TEMPI)
real_music = encode(songs[0], voc_size=VOC_SIZE, noise_level=0.2).unsqueeze(0)
probas = discriminator(real_music)
print("Proba real song", probas.item())

fig, _ = visualize_song(real_music[0], title="Real Song")
fig.savefig("real_song_with_noise.svg", format="svg")
visualize_song(fake_music[0], title="Fake Song - Untrained", highlight_max=False)
visualize_song(fake_music[0], title="Fake Song - Untrained - Max Highlighted", highlight_max=True)
plt.show()
