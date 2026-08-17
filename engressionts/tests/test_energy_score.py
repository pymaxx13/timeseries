import torch
import numpy as np
from engressionts.losses.energy_score import EnergyScoreLoss, energy_score_loss


def test_energy_score_loss_numerical_correctness():
    # 1. Generate random target and sample tensors
    torch.manual_seed(42)
    B, T, D = 4, 10, 2
    M = 15
    samples = torch.randn(M, B, T, D)
    target = torch.randn(B, T, D)
    
    # 2. Original implementation of energy_score_loss (unmasked)
    diff = samples - target.unsqueeze(0)
    term1 = torch.norm(diff, dim=-1).mean()
    pairwise = samples.unsqueeze(0) - samples.unsqueeze(1)
    term2 = torch.norm(pairwise, dim=-1).mean()
    expected_val = term1 - 0.5 * term2
    
    # 3. Call our implementation (unmasked)
    actual_val = energy_score_loss(samples, target)
    
    # 4. Compare
    assert torch.allclose(actual_val, expected_val, atol=1e-6)


def test_energy_score_loss_with_mask():
    torch.manual_seed(42)
    B, T, D = 2, 5, 1
    M = 3
    samples = torch.randn(M, B, T, D)
    target = torch.randn(B, T, D)
    mask = torch.ones(B, T, D)
    # Mask out some values
    mask[0, 2, 0] = 0.0
    mask[1, 4, 0] = 0.0
    
    # Compute masked energy score loss
    loss = energy_score_loss(samples, target, mask=mask)
    
    # Verify it ignores masked values
    # Let's manually filter out masked elements and compute standard loss
    samples_filtered = []
    target_filtered = []
    
    for b in range(B):
        for t in range(T):
            if mask[b, t, 0] == 1.0:
                samples_filtered.append(samples[:, b, t, :])  # shape [M, D]
                target_filtered.append(target[b, t, :])  # shape [D]
                
    # Stack filtered inputs:
    # samples_filtered shape: [len_active, M, D] -> [M, len_active, 1, D]
    samples_stacked = torch.stack(samples_filtered, dim=1).unsqueeze(2)  # [M, len_active, 1, D]
    target_stacked = torch.stack(target_filtered, dim=0).unsqueeze(1)  # [len_active, 1, D]
    
    expected_loss = energy_score_loss(samples_stacked, target_stacked)
    
    assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_energy_score_loss_class():
    loss_module = EnergyScoreLoss(level=[80, 95])
    assert loss_module.outputsize_multiplier == 1
    assert len(loss_module.output_names) == 5
    assert '-median' in loss_module.output_names
    
    # Update quantiles dynamically
    loss_module.update_quantile([0.1, 0.9])
    assert len(loss_module.output_names) == 2
    assert loss_module.outputsize_multiplier == 1


def test_energy_score_loss_masked_vs_unmasked_all_ones():
    # Verify that when target has D > 1, an all-ones mask gives identical values to mask=None
    torch.manual_seed(42)
    B, T, D = 4, 8, 3
    M = 10
    samples = torch.randn(M, B, T, D)
    target = torch.randn(B, T, D)
    mask = torch.ones(B, T, D)

    loss_unmasked = energy_score_loss(samples, target, mask=None)
    loss_masked = energy_score_loss(samples, target, mask=mask)

    assert torch.allclose(loss_unmasked, loss_masked, atol=1e-6)
