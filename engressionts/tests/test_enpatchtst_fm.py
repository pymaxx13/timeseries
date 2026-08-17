from unittest.mock import MagicMock, patch
import pytest
import torch
from darts.utils.likelihood_models.torch import QuantileRegression

from engressionts.models.darts.enpatchtst_fm_model import (
    EnPatchTSTFMModel,
    _EnPatchTSTFMModule,
    _PatchTSTFMBackbone,
)


@patch("engressionts.models.darts.enpatchtst_fm_model.HuggingFaceConnector")
def test_enpatchtst_fm_initialization_and_properties(mock_connector_cls):
    # Mock config returned by HF connector
    mock_connector = MagicMock()
    mock_connector.load_config.return_value = {
        "context_length": 100,
        "quantile_levels": [0.1, 0.5, 0.9],
    }
    mock_connector_cls.return_value = mock_connector

    # Initialize model
    model = EnPatchTSTFMModel(
        input_chunk_length=12,
        output_chunk_length=6,
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

    # Mock load_model behavior
    mock_model_instance = MagicMock()
    mock_connector.load_model.return_value = mock_model_instance

    # Create dummy training sample
    dummy_sample = (None, None, None, None, None, None)

    # Call _create_model
    pl_module = model._create_model(dummy_sample)

    # Verify load_model was called correctly
    mock_connector.load_model.assert_called_once()
    called_args, called_kwargs = mock_connector.load_model.call_args
    assert called_kwargs["module_class"] == _EnPatchTSTFMModule
    assert called_kwargs["pl_module_params"]["noise_std"] == 0.5
    assert called_kwargs["pl_module_params"]["noise_type"] == "uniform"
    assert called_kwargs["pl_module_params"]["num_samples"] == 15


@patch("engressionts.models.darts.enpatchtst_fm_model.HuggingFaceConnector")
def test_enpatchtst_fm_validation(mock_connector_cls):
    mock_connector = MagicMock()
    mock_connector.load_config.return_value = {
        "context_length": 20,
        "quantile_levels": [0.1, 0.5, 0.9],
    }
    mock_connector_cls.return_value = mock_connector

    # Should raise error when input_chunk_length + output_chunk_length + output_chunk_shift > context_length (20)
    with pytest.raises(ValueError, match="cannot be greater than model's maximum context_length"):
        EnPatchTSTFMModel(
            input_chunk_length=15,
            output_chunk_length=10,
        )

    # Test likelihood validation
    with pytest.raises(ValueError, match="Only QuantileRegression likelihood is supported"):
        EnPatchTSTFMModel(
            input_chunk_length=5,
            output_chunk_length=5,
            likelihood=MagicMock(),  # Not a QuantileRegression
        )

    # Quantiles must be a subset of the pre-trained ones
    with pytest.raises(ValueError, match="must be a subset of PatchTST-FM quantiles"):
        EnPatchTSTFMModel(
            input_chunk_length=5,
            output_chunk_length=5,
            likelihood=QuantileRegression(quantiles=[0.2, 0.5, 0.8]),
        )


def test_enpatchtst_fm_module_forward():
    # Initialize module with small dimensions for testing speed
    module = _EnPatchTSTFMModule(
        context_length=16,
        d_patch=4,
        d_model=8,
        n_head=2,
        n_layer=1,
        num_quantile=9,
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        noise_std=0.1,
        noise_type="gaussian",
        num_samples_train=2,
        input_chunk_length=8,
        output_chunk_length=4,
    )

    # Check that backbone is initialized
    assert isinstance(module.backbone, _PatchTSTFMBackbone)

    # Mock inputs:
    # PLModuleInput is: (x_past, x_future, x_static, future_target)
    # x_past shape: (batch_size, input_chunk_length, n_variables) -> (2, 8, 1)
    x_past = torch.randn(2, 8, 1)
    x_in = (x_past, None, None, None)

    # Run forward pass (with training=False)
    module.eval()
    with torch.no_grad():
        output = module(x_in)

    # Check output shape: (batch_size, output_chunk_length, n_variables, len(user_quantile_indices))
    # By default, likelihood=None, which resolves to median only (len = 1)
    # So expected shape is (2, 4, 1, 1)
    assert output.shape == (2, 4, 1, 1)
