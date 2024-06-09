import numpy as np
import torch
from torch.nn import functional as F

from src.config import VOC_SIZE, MIN_NOTE, MAX_NOTE


def encode(x, voc_size: int = VOC_SIZE, noise_level: float = 0.1):
    if isinstance(x, list):
        return torch.stack([encode(arr, voc_size) for arr in x])

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.clip(MIN_NOTE, MAX_NOTE)

    x = torch.nn.functional.one_hot(x - MIN_NOTE, num_classes=voc_size).float()

    return add_noise(x, noise_level)


def decode(x: torch.Tensor) -> torch.Tensor:
    return torch.argmax(x, dim=-1) + MIN_NOTE


def add_noise(x, noise_level: float):
    x = x + noise_level * torch.rand_like(x)
    return F.softmax(x.log(), dim=-1)
