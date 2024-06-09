import torch
from torch import nn
from torch.nn import functional as F


class ConvolutionalGenerator(nn.Module):
    def __init__(self, vocab_size, seq_length, voices):
        super(ConvolutionalGenerator, self).__init__()

        self.vocab_size = vocab_size  # number of unique tokens in the vocabulary
        self.seq_length = seq_length  # length of the sequence
        self.voices = voices  # number of voices in the music

        self.d_noise = 100  # dimension of the noise vector
        self.d_model = 32  # dimension of the notes embedding

        self.embedding = nn.Linear(self.d_noise, self.seq_length * self.voices * self.d_model)
        self.sequential = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.linear_ch = nn.Linear(64, 1)
        self.decoder = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        # x shape: (batch_size, d_noise)
        x = self.embedding(x)  # embed the noise vector
        x = x.reshape(-1, 1, self.seq_length, self.voices, self.d_model)

        # x shape: (batch_size, channels, seq_len, voices, d_model)
        x = self.sequential(x)  # set of convolutional layers

        # x shape: (batch_size, channels, seq_len, voices, d_model)
        x = x.permute(0, 2, 3, 4, 1)  # bring the channels to the last dimension
        x = self.linear_ch(x)
        x = F.leaky_relu(x, 0.2)
        x = x.squeeze(-1)  # remove channel dimension

        x = self.decoder(x)

        return F.softmax(x, dim=-1)
        # return F.gumbel_softmax(x, tau=1, hard=True)

    def generate(self, batch_size, device=None):
        x = torch.randn(size=(batch_size, self.d_noise), device=device)  # x shape: (batch_size, voices)
        return self.forward(x)  # probas: (batch_size, seq_len, voices, vocab_size)
