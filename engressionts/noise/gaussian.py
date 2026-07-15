import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """
    Gaussian noise injection layer.

    Adds Gaussian noise only during training.
    During evaluation, returns the input unchanged.
    """

    def __init__(self, std: float = 0.1, seed: int = 42):
        super().__init__()

        self.std = std
        self.seed = seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        if self.seed is not None:
            torch.manual_seed(self.seed)

        noise = torch.randn_like(x) * self.std

        return x + noise