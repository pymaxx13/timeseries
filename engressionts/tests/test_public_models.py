from engressionts.models.darts.block_rnn_model import EnBlockRNNModel
from engressionts.models.darts.dllinear_model import EnDLinearModel
from engressionts.models.darts.nbeats import EnBEATSModel
from engressionts.models.darts.nhits import EnHiTSModel
from engressionts.models.darts.nlinear_model import EnNLinearModel
from engressionts.models.darts.rnn_model import EnRNNModel
from engressionts.models.darts.tcn_model import EnTCNModel
from engressionts.models.darts.tft_model import EnTFTModel
from engressionts.noise import NOISE_REGISTRY


def test_public_engression_models_expose_probabilistic_defaults():
    for model_cls in (
        EnBEATSModel,
        EnHiTSModel,
        EnRNNModel,
        EnTCNModel,
        EnTFTModel,
        EnBlockRNNModel,
        EnNLinearModel,
        EnDLinearModel,
    ):
        model = model_cls(input_chunk_length=4, output_chunk_length=2)
        assert model.supports_probabilistic_prediction is True
        assert model.num_samples == 20


def test_noise_registry_contains_builtin_noise_types():
    assert "gaussian" in NOISE_REGISTRY
    assert "uniform" in NOISE_REGISTRY
