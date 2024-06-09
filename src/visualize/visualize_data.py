import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_song(song, title="Song", highlight_max=False):
    """When setting highlight=True, better visualize the actual played song: highlight the most probable pitches """
    if isinstance(song, torch.Tensor):
        song = song.detach().numpy()

    song = np.swapaxes(song, 0, 1)

    if highlight_max:
        # squeeze max towards 1 and other values towards 0
        for _ in range(5):
            song = softmax(2 * np.log(song))

    fig, axs = plt.subplots(1, song.shape[0], squeeze=False, sharex=True, sharey=True)
    axs[0, 0].set_ylabel("Pitch")
    fig.suptitle(title)
    for i, track in enumerate(song):
        axs[0, i].set_xlabel("Time")
        axs[0, i].imshow(track.T, aspect="auto", cmap='gray', origin='lower')

    return fig, axs


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
