import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import TRACKS, TEMPI, VOC_SIZE
from src.gan.discriminator import Discriminator
from src.gan.generator import ConvolutionalGenerator
from src.gan.train_utils import eval_one_epoch
from src.gan.train_utils import train_one_epoch
from src.music_player.midi_converter import create_midi_from_pitches
from src.prepare_data.load_data import get_songs
from src.prepare_data.utils import encode, decode

DATA_FOLDER = Path(__file__).parents[1] / "data"

if __name__ == "__main__":
    NUM_EPOCHS = 4
    BATCH_SIZE = 32
    LR = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999

    # DEVICE = "cpu"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    songs = get_songs(tracks=TRACKS, tempi=TEMPI)
    print(f"{len(songs)} songs loaded with {TRACKS} voices of length {TEMPI} tempi.")
    print(f"Using device: {DEVICE}. Batch size: {BATCH_SIZE}.")

    music_shape = songs[0].shape
    num_train = int(len(songs) * 0.75)
    train_loader = DataLoader(songs[:num_train], collate_fn=encode, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(songs[num_train:], collate_fn=encode, batch_size=BATCH_SIZE, shuffle=True)

    generator = ConvolutionalGenerator(vocab_size=VOC_SIZE, seq_length=TEMPI, voices=TRACKS)
    discriminator = Discriminator(vocab_size=VOC_SIZE, seq_length=TEMPI, voices=TRACKS)
    optimizer_g = Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_d = Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    criterion = nn.BCELoss()

    train_losses_g = []
    train_losses_d = []
    test_losses_g = []
    test_losses_d = []
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss_g, train_loss_d, output_d_real, output_d_fake = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            criterion=criterion,
            device=DEVICE
        )

        test_loss_g, test_loss_d = eval_one_epoch(
            generator=generator,
            discriminator=discriminator,
            test_loader=test_loader,
            criterion=criterion,
            device=DEVICE
        )

        train_losses_g.append(train_loss_g)
        train_losses_d.append(train_loss_d)
        test_losses_g.append(test_loss_g)
        test_losses_d.append(test_loss_d)

        print(f"{epoch=}: {train_loss_g=:.4e} - {train_loss_d=:.4e} | {output_d_real=:.3f} - {output_d_fake=:.3f} | "
              f"time: {time.time() - start:.2f}s")
        start = time.time()

        if epoch % 1 == 0:
            fake_song = generator.generate(1, device=DEVICE).cpu()
            create_midi_from_pitches(decode(fake_song[0]), f"_train_song_epoch_{epoch}.mid")

    # save models
    torch.save(generator.state_dict(), DATA_FOLDER / "model_generator.pth")
    torch.save(discriminator.state_dict(), DATA_FOLDER / "model_discriminator.pth")

    # plot losses
    plt.figure()
    plt.plot(train_losses_g, color='tab:red', label="Train Generator Loss")
    plt.plot(test_losses_g, color='tab:orange', label="Test Generator Loss")
    plt.plot(train_losses_d, color='tab:blue', label="Train Discriminator Loss")
    plt.plot(test_losses_d, color='tab:cyan', label="Test Discriminator Loss")

    plt.legend()
    plt.show()
