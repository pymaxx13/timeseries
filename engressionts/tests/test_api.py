import pytest
import numpy as np
import torch
from darts import TimeSeries
from engressionts.models.darts.endllinear_model import EnDLinearModel
# Or let's just use EnPatchTST which we know works well
from engressionts.models.darts.enpatchtst_fm_model import EnPatchTSTFMModel
from engressionts.models.neuralforecast.enpatchtst import EnPatchTST
import pandas as pd

@pytest.fixture
def dummy_series():
    times = pd.date_range("2000-01-01", periods=100)
    values = np.sin(np.arange(100) / 10.0) + np.random.normal(0, 0.1, 100)
    return TimeSeries.from_times_and_values(times, values)

@pytest.fixture
def dummy_df():
    times = pd.date_range("2000-01-01", periods=100)
    values = np.sin(np.arange(100) / 10.0) + np.random.normal(0, 0.1, 100)
    return pd.DataFrame({"unique_id": 1, "ds": times, "y": values})

def test_darts_num_samples_independence(dummy_series):
    # Test EnPatchTSTFMModel (Darts wrapper)
    # We will use a fast model like DLinear if possible, but let's test the interface using a mock.
    # Actually EnDLinearModel is fast. Let's use it.
    from engressionts.models.darts.endllinear_model import EnDLinearModel
    model = EnDLinearModel(input_chunk_length=12, output_chunk_length=6, num_samples=2, n_epochs=1)
    
    # Check init
    assert model.num_samples == 2
    
    model.fit(dummy_series)
    
    # Check after fit
    assert model.num_samples == 2
    
    # Predict
    forecast = model.predict(n=6, num_samples=10)
    
    # Check returned samples
    assert forecast.n_samples == 10
    
    # Check internal state did not mutate
    assert model.num_samples == 2

def test_neuralforecast_num_samples_independence(dummy_df):
    from engressionts.models.neuralforecast.enmlp import EnMLP
    from neuralforecast import NeuralForecast
    
    model = EnMLP(h=6, input_size=12, num_samples_train=2, max_steps=1)
    nf = NeuralForecast(models=[model], freq="D")
    
    assert model.num_samples_train == 2
    
    nf.fit(df=dummy_df)
    
    assert model.num_samples_train == 2
    
    # We can't pass num_samples to nf.predict easily unless it's supported by NeuralForecast
    # but we can call it on the model directly (which NeuralForecast does internally)
    # Wait, NeuralForecast's predict doesn't take num_samples natively? 
    # Actually, Engression models inside NeuralForecast don't have a public API for probabilistic prediction
    # unless they are wrapped by Darts. For NeuralForecast base models, they return mean or quantiles.
    # Let's just test that the model properties didn't mutate.
    pass

def test_stochasticity(dummy_series):
    from engressionts.models.darts.endllinear_model import EnDLinearModel
    model = EnDLinearModel(input_chunk_length=12, output_chunk_length=6, num_samples=2, n_epochs=1)
    model.fit(dummy_series)
    
    forecast = model.predict(n=6, num_samples=10)
    values = forecast.all_values()
    
    # Ensure they are not all identical
    assert np.std(values, axis=-1).mean() > 0.0

def test_clip_preds_darts(dummy_series):
    from engressionts.models.darts.endllinear_model import EnDLinearModel
    model = EnDLinearModel(input_chunk_length=12, output_chunk_length=6, num_samples=2, n_epochs=1)
    model.fit(dummy_series)
    
    # Shift series down so model predicts negative values
    shifted_series = dummy_series - 2.0
    model.fit(shifted_series)
    
    forecast_raw = model.predict(n=6, num_samples=10, clip_preds=False)
    assert np.any(forecast_raw.all_values() < 0)
    
    forecast_clipped = model.predict(n=6, num_samples=10, clip_preds=True)
    assert not np.any(forecast_clipped.all_values() < 0)
    
    # Check that positive values were unchanged
    raw_vals = forecast_raw.all_values()
    clipped_vals = forecast_clipped.all_values()
    
    # Since it's stochastic, we can't do exact equality unless we set seed
    # Let's check with seed
    forecast_raw_seed = model.predict(n=6, num_samples=10, clip_preds=False, random_state=42)
    forecast_clipped_seed = model.predict(n=6, num_samples=10, clip_preds=True, random_state=42)
    
    raw_v = forecast_raw_seed.all_values()
    clipped_v = forecast_clipped_seed.all_values()
    
    mask_pos = raw_v > 0
    np.testing.assert_allclose(raw_v[mask_pos], clipped_v[mask_pos])
    
    mask_neg = raw_v < 0
    assert np.all(clipped_v[mask_neg] == 0)
    
    # Ensure num samples doesn't change
    assert forecast_clipped.n_samples == 10
    
    # Ensure state doesn't change
    assert model.num_samples == 2
