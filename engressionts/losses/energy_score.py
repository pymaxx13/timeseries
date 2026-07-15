import torch


def energy_score_loss(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Energy Score loss.

    Parameters
    ----------
    samples
        Shape: (M, B, T, D)

    target
        Shape: (B, T, D)

    Returns
    -------
    torch.Tensor
        Scalar Energy Score loss.
    """

    # First term:
    # Distance between generated samples and ground truth
    diff = samples - target.unsqueeze(0)

    term1 = torch.norm(diff, dim=-1).mean()

    # Second term:
    # Pairwise distance between generated samples
    pairwise = samples.unsqueeze(0) - samples.unsqueeze(1)

    term2 = torch.norm(pairwise, dim=-1).mean()

    return term1 - 0.5 * term2