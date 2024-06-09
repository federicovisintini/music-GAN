import torch
import torch.nn.functional as F
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, vocab_size, seq_length, voices):
        super(Discriminator, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = 8  # 16
        self.channels_0 = 1
        self.channels_1 = 1  # 16
        self.channels_2 = 2  # 32
        self.channels_3 = 4  # 64
        self.latent_space = 8  # 128

        self.conv1 = nn.Conv3d(in_channels=self.channels_0, out_channels=self.channels_1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=self.channels_1, out_channels=self.channels_2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=self.channels_2, out_channels=self.channels_3, kernel_size=3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.fc0 = nn.Linear(vocab_size, self.d_model)
        self.linear_ch = nn.Linear(self.channels_3, 1)
        self.fc1 = nn.Linear(seq_length * voices * self.d_model, self.latent_space)
        self.fc2 = nn.Linear(self.latent_space, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, voices, vocab_size)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, seq_len, voices, vocab_size)
        x = F.leaky_relu(self.fc0(x), 0.2)  # embed the notes
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.dropout1(x)
        x = x.permute(0, 2, 3, 4, 1)  # bring the channels to the last dimension
        x = self.linear_ch(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = torch.sigmoid(self.fc2(x))

        return x
