"""End-to-end smoke test for EnHiTS on Darts' SolarDataset."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engressionts.models import EnHiTSModel


INPUT_CHUNK_LENGTH = 24
OUTPUT_CHUNK_LENGTH = 24
NUM_SAMPLES = 5


def load_solar_series() -> TimeSeries:
    """Load the Solar NIPS data using the same GluonTS path as solar-new.ipynb."""
    dataset_cache = Path(__file__).resolve().parents[2] / ".cache" / "gluonts"
    dataset = get_dataset("solar_nips", path=dataset_cache, regenerate=False)
    train_items = list(dataset.train)
    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)

    grouped_item = list(
        MultivariateGrouper(max_target_dim=target_dim)(train_items)
    )[0]
    start = grouped_item["start"].to_timestamp()
    values = np.asarray(grouped_item["target"]).T.astype(np.float32)
    times = pd.date_range(start=start, periods=values.shape[0], freq=dataset.metadata.freq)

    return TimeSeries.from_times_and_values(
        times,
        values,
        columns=[f"dim_{index}" for index in range(values.shape[1])],
    )


def main() -> None:
    series = Scaler().fit_transform(load_solar_series())
    train, validation = series.split_after(0.8)

    model = EnHiTSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        num_stacks=1,
        num_blocks=1,
        num_layers=1,
        layer_widths=32,
        pooling_kernel_sizes=((2,),),
        n_freq_downsample=((1,),),
        n_epochs=1,
        batch_size=8,
        num_samples=NUM_SAMPLES,
        random_state=42,
        pl_trainer_kwargs={
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )

    try:
        model.fit(train, val_series=validation, max_samples_per_ts=32)
        print("TRAINING: PASS")
    except Exception as error:
        print(f"TRAINING: FAIL ({error})")
        print("PREDICTION: FAIL (skipped because training failed)")
        print("PROBABILISTIC SAMPLING: FAIL (skipped because training failed)")
        raise

    try:
        prediction = model.predict(OUTPUT_CHUNK_LENGTH, num_samples=NUM_SAMPLES)
        samples = prediction.all_values(copy=False)
        print("PREDICTION: PASS")
        print(f"prediction type: {type(prediction)}")
        print(f"prediction.all_values(copy=False).shape: {samples.shape}")
    except Exception as error:
        print(f"PREDICTION: FAIL ({error})")
        print("PROBABILISTIC SAMPLING: FAIL (skipped because prediction failed)")
        raise

    samples_differ = not np.allclose(samples[:, :, 0], samples[:, :, 1])
    status = "PASS" if samples_differ else "FAIL"
    print(f"PROBABILISTIC SAMPLING: {status}")
    print(f"np.allclose(samples[:, :, 0], samples[:, :, 1]): {not samples_differ}")

    if not samples_differ:
        raise AssertionError("EnHiTS prediction samples are identical.")


if __name__ == "__main__":
    main()
