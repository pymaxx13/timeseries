from dataclasses import dataclass
from unittest.mock import MagicMock, patch
import pytest
import torch
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.models.components.timesfm2p5_submodels import (
    _ResidualBlockConfig,
    _StackedTransformersConfig,
    _TransformerConfig,
)

from engressionts.models.darts.entimesfm2p5_model import (
    EnTimesFM2p5Model,
    _EnTimesFM2p5Module,
)


@patch("engressionts.models.darts.entimesfm2p5_model.HuggingFaceConnector")
def test_entimesfm2p5_initialization_and_properties(mock_connector_cls):
    # Mock connector returned by HF connector
    mock_connector = MagicMock()
    mock_connector_cls.return_value = mock_connector

    # Initialize model
    model = EnTimesFM2p5Model(
        input_chunk_length=32,
        output_chunk_length=12,
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
    assert called_kwargs["module_class"] == _EnTimesFM2p5Module
    assert called_kwargs["pl_module_params"]["noise_std"] == 0.5
    assert called_kwargs["pl_module_params"]["noise_type"] == "uniform"
    assert called_kwargs["pl_module_params"]["num_samples"] == 15
    assert called_kwargs["additional_params"]["use_longer_projection_head"] is False


@patch("engressionts.models.darts.entimesfm2p5_model.HuggingFaceConnector")
def test_entimesfm2p5_validation(mock_connector_cls):
    mock_connector = MagicMock()
    mock_connector_cls.return_value = mock_connector

    # 1. Input/Output context limit validation (context_limit = 16384)
    # With max_icl + output_chunk_length + output_chunk_shift > 16384
    with pytest.raises(ValueError, match="cannot be greater than model's maximum context_length"):
        EnTimesFM2p5Model(
            input_chunk_length=16300,
            output_chunk_length=100,
        )

    # 2. Output limit validation (prediction_length = 128 by default)
    with pytest.raises(ValueError, match="cannot be greater than model's maximum prediction length"):
        EnTimesFM2p5Model(
            input_chunk_length=32,
            output_chunk_length=129,
            use_longer_projection_head=False,
        )

    # 3. Only QuantileRegression likelihood is supported
    with pytest.raises(ValueError, match="Only QuantileRegression likelihood is supported"):
        EnTimesFM2p5Model(
            input_chunk_length=32,
            output_chunk_length=12,
            likelihood=MagicMock(),
        )

    # 4. Quantiles must be a subset of the pre-trained ones
    with pytest.raises(ValueError, match="must be a subset of TimesFM 2.5 quantiles"):
        EnTimesFM2p5Model(
            input_chunk_length=32,
            output_chunk_length=12,
            likelihood=QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
        )


def test_entimesfm2p5_module_forward():
    @dataclass
    class TinyTimesFMConfig:
        context_limit = 100
        input_patch_len = 8
        output_patch_len = 16
        output_quantile_len = 32
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        tokenizer = _ResidualBlockConfig(
            input_dims=16,
            hidden_dims=8,
            output_dims=8,
            use_bias=True,
            activation="swish",
        )
        stacked_transformers = _StackedTransformersConfig(
            num_layers=1,
            transformer=_TransformerConfig(
                model_dims=8,
                hidden_dims=8,
                num_heads=2,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_bias=False,
                use_rotary_position_embeddings=True,
                ff_activation="swish",
                fuse_qkv=True,
            ),
        )
        output_projection_point = _ResidualBlockConfig(
            input_dims=8,
            hidden_dims=8,
            output_dims=160,  # 16 * 10
            use_bias=False,
            activation="swish",
        )
        output_projection_quantiles = _ResidualBlockConfig(
            input_dims=8,
            hidden_dims=8,
            output_dims=320,  # 32 * 10
            use_bias=False,
            activation="swish",
        )

    # Patch config class attribute
    old_config = _EnTimesFM2p5Module.config
    _EnTimesFM2p5Module.config = TinyTimesFMConfig()
    try:
        module = _EnTimesFM2p5Module(
            input_chunk_length=8,
            output_chunk_length=4,
            noise_std=0.1,
            noise_type="gaussian",
            num_samples_train=2,
            use_longer_projection_head=False,
        )

        # PLModuleInput is: (x_past, x_future, x_static, future_target)
        # x_past shape: (batch_size, input_chunk_length, n_variables) -> (2, 8, 1)
        x_past = torch.randn(2, 8, 1)
        x_in = (x_past, None, None, None)

        module.eval()
        with torch.no_grad():
            output = module(x_in)

        # Output shape should be (B, T, C, N) -> (2, 4, 1, 1)
        assert output.shape == (2, 4, 1, 1)
    finally:
        _EnTimesFM2p5Module.config = old_config
