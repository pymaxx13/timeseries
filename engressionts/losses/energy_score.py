import torch
from typing import List, Optional, Union
from neuralforecast.losses.pytorch import BasePointLoss, level_to_outputs, quantiles_to_outputs


def energy_score_loss(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes the Energy Score loss.

    Parameters
    ----------
    samples
        Shape: (M, B, T, D)

    target
        Shape: (B, T, D)

    mask
        Shape: (B, T, D) or None

    Returns
    -------
    torch.Tensor
        Scalar Energy Score loss.
    """

    if mask is None:
        # Distance between generated samples and ground truth
        diff = samples - target.unsqueeze(0)
        term1 = torch.norm(diff, dim=-1).mean()

        # Pairwise distance between generated samples
        pairwise = samples.unsqueeze(0) - samples.unsqueeze(1)
        term2 = torch.norm(pairwise, dim=-1).mean()
    else:
        # Distance between generated samples and ground truth
        diff = samples - target.unsqueeze(0)
        diff = diff * mask.unsqueeze(0)
        norm_diff = torch.norm(diff, dim=-1)
        term1 = norm_diff.sum() / torch.clamp(mask.sum() * samples.shape[0], min=1.0)

        # Pairwise distance between generated samples
        pairwise = samples.unsqueeze(0) - samples.unsqueeze(1)
        pairwise = pairwise * mask.unsqueeze(0).unsqueeze(0)
        norm_pairwise = torch.norm(pairwise, dim=-1)
        term2 = norm_pairwise.sum() / torch.clamp(mask.sum() * (samples.shape[0] ** 2), min=1.0)

    return term1 - 0.5 * term2


class EnergyScoreLoss(BasePointLoss):
    """
    Energy Score loss wrapper class for NeuralForecast compatibility.
    """
    def __init__(self, level: List[int] = [80, 90], quantiles: Optional[List[float]] = None, horizon_weight = None):
        qs, output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)
        if quantiles is not None:
            quantiles, output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)
            
        super().__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,  # Under the hood, the network outputs exactly 1 prediction channel
            output_names=output_names,
        )
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)

    def update_quantile(self, q):
        if q is not None:
            quantiles, output_names = quantiles_to_outputs(q)
            self.quantiles = torch.nn.Parameter(torch.Tensor(quantiles), requires_grad=False)
            self.output_names = output_names
            # Always ensure model configuration multiplier remains 1
            self.outputsize_multiplier = 1

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
        y_insample: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Computes Energy Score.
        In NFEngressionBaseModel, y_hat is the samples tensor of shape (M, B, H, D),
        and y is the target tensor of shape (B, H, D).
        """
        # Ensure target and prediction are at least 3D
        if y.ndim == 2:
            y = y.unsqueeze(-1)
        if y_hat.ndim == 3:
            y_hat = y_hat.unsqueeze(-1)
        if mask is not None and mask.ndim == 2:
            mask = mask.unsqueeze(-1)
            
        return energy_score_loss(samples=y_hat, target=y, mask=mask)