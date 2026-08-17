import pandas as pd
import numpy as np
import pytest
import torch

from darts import TimeSeries
from darts.models.forecasting.nf_model import NeuralForecastModel
from engressionts.models.neuralforecast.enpatchtst import EnPatchTST

def test_darts_neuralforecast_integration():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Create a toy dataset using Darts TimeSeries
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=100),
        'y': (np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)).tolist()
    })
    series = TimeSeries.from_dataframe(df, 'ds', 'y')

    # Divide into train and validation
    train, val = series[:75], series[75:]

    # 2. Instantiate NeuralForecastModel wrapping EnPatchTST
    model = NeuralForecastModel(
        input_chunk_length=24,
        output_chunk_length=24,
        model=EnPatchTST,
        model_kwargs={
            "noise_std": 0.5,
            "noise_type": "uniform",
            "num_samples_train": 7,  # Rename to num_samples_train!
            "batch_size": 16,
            "patch_len": 8,
            "stride": 4,
        },
        n_epochs=1,
        optimizer_kwargs={
            "lr": 0.001
        },
        random_state=42,
    )

    # TEST 1: Model trains successfully
    model.fit(train)
    assert model.model_created
    
    # Verify backward compatible property on inner Nixtla model (model.model.nf)
    assert model.model.nf.num_samples == 7
    assert model.model.nf.num_samples_train == 7

    # TEST 2: Normal prediction (deterministic, num_samples=1)
    pred_1 = model.predict(n=24)
    assert isinstance(pred_1, TimeSeries)
    assert len(pred_1) == 24
    assert pred_1.n_samples == 1

    # TEST 3: Probabilistic prediction (num_samples=7) does not raise error
    pred_7_1 = model.predict(n=24, num_samples=7, random_state=42)
    assert isinstance(pred_7_1, TimeSeries)
    
    # TEST 4: Verify 7 samples are present in returned TimeSeries
    assert pred_7_1.n_samples == 7
    assert pred_7_1.all_values().shape == (24, 1, 7)

    # TEST 5: Verify stochasticity (samples are not identical)
    samples = pred_7_1.all_values()  # shape (24, 1, 7)
    sample_std = np.std(samples, axis=2)  # std across samples
    # At least some time steps should have non-zero variance across the 7 samples
    assert np.any(sample_std > 1e-5), "Forecast samples are identical (deterministic) instead of stochastic!"

    # TEST 6: Verify reproducibility (repeating prediction with same seed yields identical results)
    pred_7_2 = model.predict(n=24, num_samples=7, random_state=42)
    np.testing.assert_allclose(pred_7_1.all_values(), pred_7_2.all_values(), rtol=1e-5, atol=1e-5)

    # TEST 7: Verify that predicting with different seeds yields different results
    pred_7_3 = model.predict(n=24, num_samples=7, random_state=43)
    # The trajectories should be different
    assert not np.allclose(pred_7_1.all_values(), pred_7_3.all_values(), rtol=1e-5, atol=1e-5)

    # TEST 8: Verify that num_samples=1 is deterministic (predicting twice with/without seed yields same values)
    pred_1_1 = model.predict(n=24, num_samples=1, random_state=42)
    pred_1_2 = model.predict(n=24, num_samples=1, random_state=43)
    np.testing.assert_allclose(pred_1_1.all_values(), pred_1_2.all_values(), rtol=1e-5, atol=1e-5)

    # TEST 9: Verify predicting with 100 samples yields (24, 1, 100) shape
    pred_100 = model.predict(n=24, num_samples=100, random_state=42)
    assert pred_100.n_samples == 100
    assert pred_100.all_values().shape == (24, 1, 100)

    # TEST 10: Verify deprecated `num_samples` fallback in constructor works
    model_legacy = NeuralForecastModel(
        input_chunk_length=24,
        output_chunk_length=24,
        model=EnPatchTST,
        model_kwargs={
            "noise_std": 0.5,
            "noise_type": "uniform",
            "num_samples": 12,  # legacy param
            "batch_size": 16,
            "patch_len": 8,
            "stride": 4,
        },
        n_epochs=1,
        optimizer_kwargs={"lr": 0.001},
        random_state=42,
    )
    model_legacy.fit(train)
    assert model_legacy.model.nf.num_samples == 12
    assert model_legacy.model.nf.num_samples_train == 12
    
    print("All integration tests passed successfully!")
