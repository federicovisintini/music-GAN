import torch
from torch import nn
from torch.nn import functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-Head Self Attention
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed Forward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (seq_len, batch_size, voices, d_model)
        seq_len, batch_size, voices, d_model = x.shape

        # Flatten the voices dimension into the batch dimension for the attention layer
        x = x.reshape(seq_len, batch_size * voices, d_model)

        # Masked Multi-Head Attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)

        # Apply dropout to the attention output
        attn_output = self.dropout1(attn_output)

        # Add & Layer Norm
        x = x + attn_output
        x = self.norm1(x)

        # Feed Forward Network
        ff_output = self.feedforward(x)

        # Apply dropout to the feedforward output
        ff_output = self.dropout2(ff_output)

        # Add & Layer Norm
        x = x + ff_output
        x = self.norm2(x)

        # Reshape back to the original shape (seq_len, batch_size, voices, d_model)
        x = x.view(seq_len, batch_size, voices, d_model)

        return x


class DCGANGenerator(nn.Module):
    def __init__(self, vocab_size, seq_length, voices):
        """Note vocab_size must be 128 !"""
        super(DCGANGenerator, self).__init__()

        self.vocab_size = vocab_size  # number of unique tokens in the vocabulary
        self.seq_length = seq_length  # length of the sequence
        self.voices = voices  # number of voices in the music

        self.d_noise = 100  # dimension of the noise vector
        self.d_model = 32  # max_voices - check dcgan

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.d_noise, self.d_model * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.d_model * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(self.d_model * 16, self.d_model * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_model * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(self.d_model * 8, self.d_model * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_model * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(self.d_model * 4, self.d_model * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_model * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(self.d_model * 2, self.d_model, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ngf) x 64 x 64``
            nn.ConvTranspose2d(self.d_model, self.voices, 4, 2, 1, bias=False),
            nn.Linear(128, self.seq_length),
            nn.Tanh()
            # state size. ``voices x 128 x seq_length``
        )

    def forward(self, x):
        """For DC-GAN"""
        x = x.unsqueeze(-1).unsqueeze(.1)
        x = self.main(x)
        x.permute(0, 3, 1, 2)
        return F.softmax(x, dim=-1)

    def generate(self, batch_size, device=None):
        x = torch.randn(size=(batch_size, self.d_noise, 1, 1), device=device)  # x shape: (batch_size, voices)
        return self.forward(x)  # probas: (batch_size, seq_len, voices, vocab_size)


class MMM(nn.Module):
    def __init__(self, vocab_size, seq_length, voices):
        super(MMM, self).__init__(vocab_size, seq_length, voices)

        self.vocab_size = vocab_size  # number of unique tokens in the vocabulary
        self.seq_length = seq_length  # length of the sequence
        self.voices = voices  # number of voices in the music

        self.d_model = 8  # dimension of the notes embedding
        self.d_ff = 128  # dimension of the feedforward network
        self.nhead = 4  # number of heads in the multi-head attention
        self.nblocks = 4  # number of transformer blocks

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.seq_length, self.voices, self.d_model))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_ff)
            for _ in range(self.nblocks)
        ])

        self.fc_out = nn.Linear(self.d_model, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, voices)
        x = self.embedding(x) + self.positional_encoding

        # x shape: (batch_size, seq_len, voices, d_model)
        x = x.transpose(1, 0)

        for block in self.blocks:
            x = block(x, mask=mask)

        # x shape: (seq_len, batch_size, voices, d_model)
        x = self.fc_out(x)

        # x shape: (seq_len, batch_size, voices, vocab_size)
        x = x.transpose(1, 0)

        return x

    def generate(self, batch_size):
        x = self.generate_random_noise(batch_size=batch_size)  # x shape: (batch_size, 1, voices)

        # create an empty tensor to store the generated sequence
        generated = torch.zeros((batch_size, self.seq_length, self.voices), dtype=torch.long)
        generated[:, 0, :] = x.squeeze(1)

        trues = torch.ones((self.seq_length, self.seq_length), dtype=torch.bool)
        causal_mask = torch.triu(trues, diagonal=1).to(x.device)

        for pos in range(1, batch_size):
            logits = self.forward(generated, mask=causal_mask)  # logits: (batch_size, seq_len, voices, vocab_size)
            next_token = torch.argmax(self.softmax(logits[:, pos, :, :]), dim=-1).unsqueeze(1)
            generated[:, pos, :] = next_token.squeeze(1)

        return generated

    def generate_random_noise(self, batch_size: int):
        return torch.randint(low=0, high=self.vocab_size, size=(batch_size, 1, self.voices))
