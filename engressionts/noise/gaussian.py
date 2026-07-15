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
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
            if self.seed is not None:
                self._generator.manual_seed(self.seed)
        return self._generator

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the local noise stream used by reproducible evaluations."""
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            raise ValueError("No seed is configured for this noise layer.")
        if self._generator is not None:
            self._generator.manual_seed(self.seed)

    def reset_std(self, std: float) -> None:
        """Update the noise scale for an evaluation run."""
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=self._get_generator(x.device),
        ) * self.std

        return x + noise
