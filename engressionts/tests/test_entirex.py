import sys
from unittest.mock import MagicMock, patch
import pytest
import torch
from darts.utils.likelihood_models.torch import QuantileRegression

# Create a mock for tirex since it's not installed in our env
sys.modules["tirex"] = MagicMock()
sys.modules["tirex.models"] = MagicMock()
sys.modules["tirex.models.tirex"] = MagicMock()

from engressionts.models.darts.entirex_model import (
    EnTiRexModel,
    _EnTiRexModule,
)


@patch("engressionts.models.darts.entirex_model.load_model")
def test_entirex_initialization_and_properties(mock_load_model):
    # Setup mock TiRex model
    mock_tirex = MagicMock()
    mock_load_model.return_value = mock_tirex

    # Initialize model (must pass accept_license=True)
    model = EnTiRexModel(
        input_chunk_length=12,
        output_chunk_length=6,
        accept_license=True,
        noise_std=0.5,
        noise_type="uniform",
        num_samples=15,
    )

    # Verify properties
    assert model.supports_probabilistic_prediction is True
    assert model.supports_past_covariates is False
    assert model.supports_future_covariates is False
    assert model.noise_std == 0.5
    assert model.noise_type == "uniform"
    assert model.num_samples == 15
    assert model.num_samples_train == 15

    # Create dummy training sample
    dummy_sample = (None, None, None, None, None, None)

    # Call _create_model
    pl_module = model._create_model(dummy_sample)

    # Verify load_model was called correctly
    mock_load_model.assert_called_once()
    assert pl_module.noise_std == 0.5
    assert pl_module.noise_type == "uniform"
    assert pl_module.num_samples == 15


@patch("engressionts.models.darts.entirex_model.load_model")
def test_entirex_validation(mock_load_model):
    mock_tirex = MagicMock()
    mock_load_model.return_value = mock_tirex

    # Validation: license accept is required
    with pytest.raises(ValueError, match="TiRex is distributed under the NXAI Community License"):
        EnTiRexModel(
            input_chunk_length=12,
            output_chunk_length=6,
            accept_license=False,
        )

    # Validation: max prediction length (2048)
    with pytest.raises(ValueError, match="cannot be greater than model's maximum prediction length"):
        EnTiRexModel(
            input_chunk_length=12,
            output_chunk_length=2049,
            accept_license=True,
        )

    # Validation: Only QuantileRegression likelihood is supported
    with pytest.raises(ValueError, match="Only QuantileRegression likelihood is supported"):
        EnTiRexModel(
            input_chunk_length=12,
            output_chunk_length=6,
            accept_license=True,
            likelihood=MagicMock(),
        )

    # Validation: Quantiles must be a subset of the pre-trained ones
    with pytest.raises(ValueError, match="must be a subset of TiRex quantiles"):
        EnTiRexModel(
            input_chunk_length=12,
            output_chunk_length=6,
            accept_license=True,
            likelihood=QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
        )


@patch("engressionts.models.darts.entirex_model.load_model")
def test_entirex_module_forward(mock_load_model):
    # Setup mock TiRex model
    mock_tirex = MagicMock()
    # Mock return values for forecast
    mock_quantiles = torch.randn(2, 10, 9)
    mock_tirex._forecast_quantiles.return_value = (mock_quantiles, {})
    mock_load_model.return_value = mock_tirex

    # Initialize module
    module = _EnTiRexModule(
        tirex_kwargs={},
        all_quantiles=EnTiRexModel._DEFAULT_QUANTILES,
        noise_std=0.1,
        noise_type="gaussian",
        num_samples_train=2,
        input_chunk_length=8,
        output_chunk_length=6,
        output_chunk_shift=4,  # future_len = 10
    )

    # Mock inputs:
    # PLModuleInput is: (x_past, x_future, x_static, future_target)
    # x_past shape: (batch_size, input_chunk_length, n_variables) -> (2, 8, 1)
    x_past = torch.randn(2, 8, 1)
    x_in = (x_past, None, None, None)

    module.eval()
    with torch.no_grad():
        output = module(x_in)

    # Verify that load_model was called and setup correctly
    mock_load_model.assert_called_once()
    assert mock_tirex._forecast_quantiles.call_count == 1
    called_kwargs = mock_tirex._forecast_quantiles.call_args[1]
    assert called_kwargs["context"].shape == (2, 8)
    assert called_kwargs["prediction_length"] == 10
    assert called_kwargs["output_device"] == str(x_past.device)

    # Output shape: (B, T, C, N) -> (2, 6, 1, 1)
    assert output.shape == (2, 6, 1, 1)
