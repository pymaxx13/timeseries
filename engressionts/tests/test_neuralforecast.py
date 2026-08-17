import pandas as pd
import numpy as np
import pytest
import torch
import torch.nn as nn
from neuralforecast import NeuralForecast
from engressionts.base.base_engression import NFEngressionBaseModel
from engressionts.losses.energy_score import EnergyScoreLoss


class MockEnModel(NFEngressionBaseModel):
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    EXOGENOUS_CAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        loss=None,
        valid_loss=None,
        noise_std=1.0,
        noise_type="gaussian",
        num_samples=20,
        **kwargs
    ):
        if loss is None:
            loss = EnergyScoreLoss()
        if valid_loss is None:
            valid_loss = EnergyScoreLoss()
            
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
            valid_loss=valid_loss,
            noise_std=noise_std,
            noise_type=noise_type,
            num_samples=num_samples,
            **kwargs
        )
        self.linear = nn.Linear(input_size, h)

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]  # [B * M, L, 1]
        insample_y = self.noise_layer(insample_y)
        insample_y = insample_y.squeeze(-1)
        out = self.linear(insample_y)
        return out.unsqueeze(-1)


def test_nf_engression_base_model_defaults():
    model = MockEnModel(h=3, input_size=6)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_nf_engression_base_model_fit_and_predict():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Create a toy dataset
    df = pd.DataFrame({
        'unique_id': [1] * 20 + [2] * 20,
        'ds': pd.date_range(start='2020-01-01', periods=20).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 40)) + np.random.normal(0, 0.1, 40)).tolist()
    })

    # 2. Fit the model via NeuralForecast wrapper
    model = MockEnModel(
        h=3,
        input_size=6,
        max_steps=10,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=15,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    # 3. Prediction
    fcst = nf.predict()
    
    # 4. Check output columns and shapes
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'MockEnModel-median',
        'MockEnModel-lo-80',
        'MockEnModel-hi-80',
        'MockEnModel-lo-90',
        'MockEnModel-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_patchtst_defaults():
    from engressionts.models.neuralforecast.enpatchtst import EnPatchTST
    model = EnPatchTST(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_patchtst_fit_and_predict():
    from engressionts.models.neuralforecast.enpatchtst import EnPatchTST
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnPatchTST(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
        patch_len=8,
        stride=4,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnPatchTST-median',
        'EnPatchTST-lo-80',
        'EnPatchTST-hi-80',
        'EnPatchTST-lo-90',
        'EnPatchTST-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_autoformer_defaults():
    from engressionts.models.neuralforecast.enautoformer import EnAutoformer
    model = EnAutoformer(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_autoformer_fit_and_predict():
    from engressionts.models.neuralforecast.enautoformer import EnAutoformer
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnAutoformer(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnAutoformer-median',
        'EnAutoformer-lo-80',
        'EnAutoformer-hi-80',
        'EnAutoformer-lo-90',
        'EnAutoformer-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_informer_defaults():
    from engressionts.models.neuralforecast.eninformer import EnInformer
    model = EnInformer(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_informer_fit_and_predict():
    from engressionts.models.neuralforecast.eninformer import EnInformer
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnInformer(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnInformer-median',
        'EnInformer-lo-80',
        'EnInformer-hi-80',
        'EnInformer-lo-90',
        'EnInformer-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_fedformer_defaults():
    from engressionts.models.neuralforecast.enfedformer import EnFEDformer
    model = EnFEDformer(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_fedformer_fit_and_predict():
    from engressionts.models.neuralforecast.enfedformer import EnFEDformer
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnFEDformer(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnFEDformer-median',
        'EnFEDformer-lo-80',
        'EnFEDformer-hi-80',
        'EnFEDformer-lo-90',
        'EnFEDformer-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_itransformer_defaults():
    from engressionts.models.neuralforecast.enitransformer import EniTransformer
    model = EniTransformer(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_itransformer_fit_and_predict():
    from engressionts.models.neuralforecast.enitransformer import EniTransformer
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EniTransformer(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EniTransformer-median',
        'EniTransformer-lo-80',
        'EniTransformer-hi-80',
        'EniTransformer-lo-90',
        'EniTransformer-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_timexer_defaults():
    from engressionts.models.neuralforecast.entimexer import EnTimeXer
    model = EnTimeXer(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_timexer_fit_and_predict():
    from engressionts.models.neuralforecast.entimexer import EnTimeXer
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnTimeXer(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnTimeXer-median',
        'EnTimeXer-lo-80',
        'EnTimeXer-hi-80',
        'EnTimeXer-lo-90',
        'EnTimeXer-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_timesnet_defaults():
    from engressionts.models.neuralforecast.entimesnet import EnTimesNet
    model = EnTimesNet(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_timesnet_fit_and_predict():
    from engressionts.models.neuralforecast.entimesnet import EnTimesNet
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnTimesNet(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnTimesNet-median',
        'EnTimesNet-lo-80',
        'EnTimesNet-hi-80',
        'EnTimesNet-lo-90',
        'EnTimesNet-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6





def test_en_tsmixerx_defaults():
    from engressionts.models.neuralforecast.entsmixerx import EnTSMixerx
    model = EnTSMixerx(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_tsmixerx_fit_and_predict():
    from engressionts.models.neuralforecast.entsmixerx import EnTSMixerx
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnTSMixerx(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnTSMixerx-median',
        'EnTSMixerx-lo-80',
        'EnTSMixerx-hi-80',
        'EnTSMixerx-lo-90',
        'EnTSMixerx-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_mlp_defaults():
    from engressionts.models.neuralforecast.enmlp import EnMLP
    model = EnMLP(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_mlp_fit_and_predict():
    from engressionts.models.neuralforecast.enmlp import EnMLP
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnMLP(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnMLP-median',
        'EnMLP-lo-80',
        'EnMLP-hi-80',
        'EnMLP-lo-90',
        'EnMLP-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_mlpmultivariate_defaults():
    from engressionts.models.neuralforecast.enmlpmultivariate import EnMLPMultivariate
    model = EnMLPMultivariate(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_mlpmultivariate_fit_and_predict():
    from engressionts.models.neuralforecast.enmlpmultivariate import EnMLPMultivariate
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnMLPMultivariate(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnMLPMultivariate-median',
        'EnMLPMultivariate-lo-80',
        'EnMLPMultivariate-hi-80',
        'EnMLPMultivariate-lo-90',
        'EnMLPMultivariate-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_kan_defaults():
    from engressionts.models.neuralforecast.enkan import EnKAN
    model = EnKAN(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_kan_fit_and_predict():
    from engressionts.models.neuralforecast.enkan import EnKAN
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnKAN(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnKAN-median',
        'EnKAN-lo-80',
        'EnKAN-hi-80',
        'EnKAN-lo-90',
        'EnKAN-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_bitcn_defaults():
    from engressionts.models.neuralforecast.enbitcn import EnBiTCN
    model = EnBiTCN(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_bitcn_fit_and_predict():
    from engressionts.models.neuralforecast.enbitcn import EnBiTCN
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnBiTCN(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnBiTCN-median',
        'EnBiTCN-lo-80',
        'EnBiTCN-hi-80',
        'EnBiTCN-lo-90',
        'EnBiTCN-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_xlinear_defaults():
    from engressionts.models.neuralforecast.enxlinear import EnXLinear
    model = EnXLinear(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_xlinear_fit_and_predict():
    from engressionts.models.neuralforecast.enxlinear import EnXLinear
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnXLinear(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnXLinear-median',
        'EnXLinear-lo-80',
        'EnXLinear-hi-80',
        'EnXLinear-lo-90',
        'EnXLinear-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_xlstm_defaults():
    from engressionts.models.neuralforecast.enxlstm import EnxLSTM, IS_XLSTM_INSTALLED
    if not IS_XLSTM_INSTALLED:
        pytest.skip("xlstm package not installed")
    model = EnxLSTM(h=3, input_size=16)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_xlstm_fit_and_predict():
    from engressionts.models.neuralforecast.enxlstm import EnxLSTM, IS_XLSTM_INSTALLED
    if not IS_XLSTM_INSTALLED:
        pytest.skip("xlstm package not installed")
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnxLSTM(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnxLSTM-median',
        'EnxLSTM-lo-80',
        'EnxLSTM-hi-80',
        'EnxLSTM-lo-90',
        'EnxLSTM-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_softs_defaults():
    from engressionts.models.neuralforecast.ensofts import EnSOFTS
    model = EnSOFTS(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_softs_fit_and_predict():
    from engressionts.models.neuralforecast.ensofts import EnSOFTS
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnSOFTS(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnSOFTS-median',
        'EnSOFTS-lo-80',
        'EnSOFTS-hi-80',
        'EnSOFTS-lo-90',
        'EnSOFTS-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_softssharp_defaults():
    from engressionts.models.neuralforecast.ensoftssharp import EnSOFTSSharp
    model = EnSOFTSSharp(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_softssharp_fit_and_predict():
    from engressionts.models.neuralforecast.ensoftssharp import EnSOFTSSharp
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnSOFTSSharp(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnSOFTSSharp-median',
        'EnSOFTSSharp-lo-80',
        'EnSOFTSSharp-hi-80',
        'EnSOFTSSharp-lo-90',
        'EnSOFTSSharp-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_stemgnn_defaults():
    from engressionts.models.neuralforecast.enstemgnn import EnStemGNN
    model = EnStemGNN(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_stemgnn_fit_and_predict():
    from engressionts.models.neuralforecast.enstemgnn import EnStemGNN
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnStemGNN(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnStemGNN-median',
        'EnStemGNN-lo-80',
        'EnStemGNN-hi-80',
        'EnStemGNN-lo-90',
        'EnStemGNN-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6


def test_en_rmok_defaults():
    from engressionts.models.neuralforecast.enrmok import EnRMoK
    model = EnRMoK(h=3, input_size=16, n_series=2)
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_rmok_fit_and_predict():
    from engressionts.models.neuralforecast.enrmok import EnRMoK
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 2,
        'y': (np.sin(np.linspace(0, 10, 60)) + np.random.normal(0, 0.1, 60)).tolist()
    })

    model = EnRMoK(
        h=3,
        input_size=16,
        n_series=2,
        max_steps=5,
        learning_rate=1e-3,
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnRMoK-median',
        'EnRMoK-lo-80',
        'EnRMoK-hi-80',
        'EnRMoK-lo-90',
        'EnRMoK-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 6





def test_en_hint_defaults():
    from engressionts.models.neuralforecast.enhint import EnHINT
    from engressionts.models.neuralforecast.enmlp import EnMLP
    
    S = np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    base_model = EnMLP(h=3, input_size=16)
    model = EnHINT(h=3, S=S, model=base_model, reconciliation="BottomUp")
    assert model.num_samples == 20
    assert model.noise_std == 1.0
    assert model.noise_type == "gaussian"
    assert model.loss.__class__.__name__ == "EnergyScoreLoss"


def test_en_hint_fit_and_predict():
    from engressionts.models.neuralforecast.enhint import EnHINT
    from engressionts.models.neuralforecast.enmlp import EnMLP
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        'unique_id': [1] * 30 + [2] * 30 + [3] * 30,
        'ds': pd.date_range(start='2020-01-01', periods=30).tolist() * 3,
        'y': (np.sin(np.linspace(0, 10, 90)) + np.random.normal(0, 0.1, 90)).tolist()
    })

    S = np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    base_model = EnMLP(
        h=3,
        input_size=16,
        max_steps=5,
        learning_rate=1e-3,
    )
    
    model = EnHINT(
        h=3,
        S=S,
        model=base_model,
        reconciliation="BottomUp",
        noise_std=0.5,
        num_samples=10,
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df, val_size=3)

    fcst = nf.predict()
    
    assert isinstance(fcst, pd.DataFrame)
    assert 'unique_id' in fcst.columns
    assert 'ds' in fcst.columns
    
    expected_cols = [
        'EnHINT-median',
        'EnHINT-lo-80',
        'EnHINT-hi-80',
        'EnHINT-lo-90',
        'EnHINT-hi-90',
    ]
    for col in expected_cols:
        assert col in fcst.columns, f"Expected column {col} missing in prediction"
        
    assert len(fcst) == 9






















