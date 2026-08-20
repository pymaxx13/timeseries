# ============================================================
# CELL 1 — INSTALL DEPENDENCIES
# ============================================================

!pip install -q "u8darts[torch]" gluonts lightning

# --- CELL ---

# ============================================================
# CELL 2 — IMPORTS
# ============================================================

import os
import gc
import time
import random
import warnings
import logging
import inspect

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --- CELL ---

# ============================================================
# CELL 3 — DETERMINISTIC SEEDS
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

print("Seed:", SEED)

# --- CELL ---

# ============================================================
# CELL 4 — DARTS IMPORTS
# ============================================================

from darts import TimeSeries

from darts.models import (
    NBEATSModel,
    NHiTSModel,
    TCNModel,
    TFTModel,
    TransformerModel,
    RNNModel,
    BlockRNNModel,
    DLinearModel,
    NLinearModel,
    TiDEModel,
    TSMixerModel,
    Chronos2Model,
    TimesFM2p5Model,
)

from darts.models import (
    ConformalNaiveModel,
    ConformalQRModel,
)

from darts.utils.likelihood_models import (
    GaussianLikelihood,
    QuantileRegression,
)

print("Darts imports successful.")

# --- CELL ---

# ============================================================
# CELL 5 — EXPERIMENT CONFIGURATION
# ============================================================

DATASET_NAME = "solar_nips"

PRED_LEN = 24

LAGS = (1, 24, 168)

NUM_NODES = 137

NUM_PRED_SAMPLES = 100

ALPHA = 0.05

POINT_METHOD = "median"

QUANTILES = [
    0.1,
    0.5,
    0.9
]

# Separate calibration region for conformal prediction.
# 168 hours = 7 days for hourly Solar NIPS data.
CAL_LENGTH = 168

# Training epochs
N_EPOCHS = 1

BATCH_SIZE = 64

LEARNING_RATE = 1e-3

print("Prediction horizon:", PRED_LEN)
print("Calibration length:", CAL_LENGTH)
print("Epochs:", N_EPOCHS)

# --- CELL ---

# ============================================================
# CELL 6 — LOAD GLUONTS SOLAR NIPS
# ============================================================

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

ds = get_dataset(
    DATASET_NAME,
    regenerate=False
)

print(ds.metadata)
print("Frequency:", ds.metadata.freq)

# --- CELL ---

# ============================================================
# CELL 7 — BUILD DARTS MULTIVARIATE TIMESERIES
# ============================================================

def gluonts_item_to_darts_mv(item, freq: str) -> TimeSeries:
    start = item["start"].to_timestamp() if hasattr(item["start"], "to_timestamp") else pd.Timestamp(item["start"])
    target = np.asarray(item["target"])
    if target.ndim != 2:
        raise ValueError(f"Expected multivariate target with ndim=2, got shape {target.shape}")
    
    values = target.T
    times = pd.date_range(start=start, periods=values.shape[0], freq=freq)
    cols = [f"dim_{i}" for i in range(values.shape[1])]
    return TimeSeries.from_times_and_values(times, values, columns=cols)

freq = ds.metadata.freq
target_dim = int(ds.metadata.feat_static_cat[0].cardinality)
train_grouper = MultivariateGrouper(max_target_dim=target_dim)
train_mv_items = list(train_grouper(list(ds.train)))
train_ts = gluonts_item_to_darts_mv(train_mv_items[0], freq)

print("Train TS length:", len(train_ts))


# --- CELL ---

# ============================================================
# CELL 8 — SCALING
# ============================================================

from darts.dataprocessing.transformers import Scaler

y_scaler = Scaler()

# IMPORTANT: keep the same variable name used everywhere below
train_y_sc = y_scaler.fit_transform(train_ts)

print("Scaled train shape:", train_y_sc.shape)

# --- CELL ---

# ============================================================
# CELL 9 — COVARIATE FUNCTIONS
# ============================================================

from darts import concatenate


def lag_covs_from_scaled_target(
    ts_sc,
    lags=(1, 24, 168)
):
    shifted = []

    for L in lags:

        s = ts_sc.shift(L)

        s = s.with_columns_renamed(
            ts_sc.components,
            [
                f"{c}_lag{L}"
                for c in ts_sc.components
            ]
        )

        shifted.append(s)

    common = shifted[0]

    for s in shifted[1:]:
        common = common.slice_intersect(s)

    shifted = [
        s.slice_intersect(common)
        for s in shifted
    ]

    return concatenate(
        shifted,
        axis=1
    )


def fourier_from_index(idx):

    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    X = np.vstack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
    ]).T

    return TimeSeries.from_times_and_values(
        idx,
        X,
        columns=[
            "h_sin",
            "h_cos",
            "dow_sin",
            "dow_cos"
        ]
    )


def dim_indicator_norm(idx, D):

    if D == 1:
        v = np.zeros(1, dtype=np.float32)
    else:
        v = (
            np.arange(D, dtype=np.float32)
            / (D - 1)
        )

    X = np.tile(
        v,
        (len(idx), 1)
    )

    cols = [
        f"dim_id_{i}"
        for i in range(D)
    ]

    return TimeSeries.from_times_and_values(
        idx,
        X,
        columns=cols
    )


def build_past_covs(
    ts_sc,
    lags=(1, 24, 168)
):

    lag_covs = lag_covs_from_scaled_target(
        ts_sc,
        lags
    )

    idx = lag_covs.time_index

    time_covs = fourier_from_index(idx)

    dim_covs = dim_indicator_norm(
        idx,
        ts_sc.width
    )

    return concatenate(
        [
            lag_covs,
            dim_covs,
            time_covs
        ],
        axis=1
    )

# --- CELL ---

# ============================================================
# CELL 10 — TRAINING COVARIATES
# ============================================================

train_pc = build_past_covs(
    train_y_sc,
    lags=LAGS
)

train_y_sc = train_y_sc.slice_intersect(
    train_pc
)

train_pc = train_pc.slice_intersect(
    train_y_sc
)

train_pc = train_pc.astype(np.float32)
train_y_sc = train_y_sc.astype(np.float32)

print("Train target:", train_y_sc.shape)
print("Train covariates:", train_pc.shape)

# --- CELL ---

# ============================================================
# CELL 11 — TRAIN / CALIBRATION SPLIT
# ============================================================

cal_end = train_y_sc.end_time()

cal_start = (
    cal_end
    - pd.Timedelta(hours=CAL_LENGTH - 1)
)

# Calibration target
calibration_y_sc = train_y_sc[
    cal_start:
]

# Actual model-training target:
# everything BEFORE calibration region
base_train_y_sc = train_y_sc[
    :cal_start - pd.Timedelta(hours=1)
]

# Corresponding covariates
base_train_pc = train_pc[
    :base_train_y_sc.end_time()
]

calibration_pc = train_pc[
    cal_start:
]

print(
    "Base training:",
    base_train_y_sc.start_time(),
    "->",
    base_train_y_sc.end_time()
)

print(
    "Calibration:",
    calibration_y_sc.start_time(),
    "->",
    calibration_y_sc.end_time()
)

print("Base train shape:", base_train_y_sc.shape)
print("Calibration shape:", calibration_y_sc.shape)

# --- CELL ---

# ============================================================
# CELL 12 — CREATE TEST WINDOWS
# ============================================================

def get_test_windows(
    dataset,
    num_nodes=137
):

    all_series = []

    for entry in dataset:

        idx = pd.date_range(
            start=entry["start"].to_timestamp(),
            periods=len(entry["target"]),
            freq=entry["start"].freqstr
        )

        all_series.append(
            pd.Series(
                entry["target"],
                index=idx
            )
        )

    num_windows = (
        len(all_series)
        // num_nodes
    )

    windows = []

    for w in range(num_windows):

        start_idx = (
            w * num_nodes
        )

        end_idx = (
            (w + 1) * num_nodes
        )

        window_df = pd.concat(
            all_series[
                start_idx:end_idx
            ],
            axis=1
        )

        window_df.columns = [
            f"node_{i}"
            for i in range(num_nodes)
        ]

        windows.append(window_df)

    return windows


test_windows = get_test_windows(
    ds.test,
    num_nodes=NUM_NODES
)

print(
    "Number of test windows:",
    len(test_windows)
)

# --- CELL ---

# ============================================================
# CELL 13 — METRIC FUNCTIONS
# ============================================================

def _sync_tensors(*args):

    tensors = [
        torch.as_tensor(x)
        if not isinstance(x, torch.Tensor)
        else x
        for x in args
    ]

    device = tensors[0].device

    return [
        t.to(
            device,
            dtype=torch.float32
        )
        for t in tensors
    ]


def get_point_forecast(
    y_pred,
    method="median"
):

    if method == "mean":
        return torch.mean(
            y_pred,
            dim=-1
        )

    if method == "median":
        return torch.median(
            y_pred,
            dim=-1
        ).values

    if isinstance(method, float):
        return torch.quantile(
            y_pred,
            method,
            dim=-1
        )

    raise ValueError(
        "Invalid point method"
    )


def aggregate(x):
    return torch.mean(x)


def metric_mae(
    y_true,
    y_pred
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    yp = get_point_forecast(
        y_pred,
        POINT_METHOD
    )

    return aggregate(
        torch.abs(
            y_true - yp
        )
    ).item()


def metric_mse(
    y_true,
    y_pred
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    yp = get_point_forecast(
        y_pred,
        POINT_METHOD
    )

    return aggregate(
        (y_true - yp) ** 2
    ).item()


def metric_rmse(
    y_true,
    y_pred
):

    return np.sqrt(
        metric_mse(
            y_true,
            y_pred
        )
    )


def metric_mape(
    y_true,
    y_pred,
    eps=1e-8
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    yp = get_point_forecast(
        y_pred,
        POINT_METHOD
    )

    value = torch.abs(
        (y_true - yp)
        / torch.clamp(
            torch.abs(y_true),
            min=eps
        )
    )

    return (
        torch.mean(value) * 100
    ).item()


def metric_smape(
    y_true,
    y_pred,
    eps=1e-8
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    yp = get_point_forecast(
        y_pred,
        POINT_METHOD
    )

    numerator = torch.abs(
        yp - y_true
    )

    denominator = torch.clamp(
        (
            torch.abs(y_true)
            + torch.abs(yp)
        ) / 2,
        min=eps
    )

    return (
        torch.mean(
            numerator / denominator
        ) * 100
    ).item()


def metric_mase(
    y_true,
    y_pred,
    y_train,
    eps=1e-8
):

    y_true, y_pred, y_train = _sync_tensors(
        y_true,
        y_pred,
        y_train
    )

    yp = get_point_forecast(
        y_pred,
        POINT_METHOD
    )

    mae_forecast = torch.abs(
        y_true - yp
    )

    diff = torch.abs(
        y_train[1:]
        - y_train[:-1]
    )

    scale = torch.mean(
        diff,
        dim=0
    )

    scale = torch.clamp(
        scale,
        min=eps
    )

    value = (
        mae_forecast
        / scale.unsqueeze(0)
    )

    return torch.mean(
        value
    ).item()


def metric_rmsse(
    y_true,
    y_pred,
    y_train,
    eps=1e-8
):

    y_true, y_pred, y_train = _sync_tensors(
        y_true,
        y_pred,
        y_train
    )

    yp = get_point_forecast(
        y_pred,
        POINT_METHOD
    )

    mse_forecast = (
        y_true - yp
    ) ** 2

    diff = (
        y_train[1:]
        - y_train[:-1]
    ) ** 2

    scale = torch.mean(
        diff,
        dim=0
    )

    scale = torch.clamp(
        scale,
        min=eps
    )

    value = torch.sqrt(
        mse_forecast
        / scale.unsqueeze(0)
    )

    return torch.mean(
        value
    ).item()


def metric_crps(
    y_true,
    y_pred
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    y_true_ext = (
        y_true.unsqueeze(-1)
    )

    abs_diff_true = torch.mean(
        torch.abs(
            y_pred
            - y_true_ext
        ),
        dim=-1
    )

    y_i = y_pred.unsqueeze(-1)
    y_j = y_pred.unsqueeze(-2)

    abs_diff_samples = torch.mean(
        torch.abs(
            y_i - y_j
        ),
        dim=(-1, -2)
    )

    score = (
        abs_diff_true
        - 0.5 * abs_diff_samples
    )

    return torch.mean(score).item()


def metric_picp(
    y_true,
    y_pred,
    alpha=0.05
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    lower = torch.quantile(
        y_pred,
        alpha / 2,
        dim=-1
    )

    upper = torch.quantile(
        y_pred,
        1 - alpha / 2,
        dim=-1
    )

    inside = (
        (y_true >= lower)
        & (y_true <= upper)
    ).float()

    return torch.mean(
        inside
    ).item()


def metric_mpiw(
    y_pred,
    alpha=0.05
):

    y_pred = _sync_tensors(
        y_pred
    )[0]

    lower = torch.quantile(
        y_pred,
        alpha / 2,
        dim=-1
    )

    upper = torch.quantile(
        y_pred,
        1 - alpha / 2,
        dim=-1
    )

    return torch.mean(
        upper - lower
    ).item()


def metric_mis(
    y_true,
    y_pred,
    alpha=0.05
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    lower = torch.quantile(
        y_pred,
        alpha / 2,
        dim=-1
    )

    upper = torch.quantile(
        y_pred,
        1 - alpha / 2,
        dim=-1
    )

    widths = upper - lower

    below = (
        y_true < lower
    ).float()

    above = (
        y_true > upper
    ).float()

    penalty_below = (
        (2 / alpha)
        * (lower - y_true)
        * below
    )

    penalty_above = (
        (2 / alpha)
        * (y_true - upper)
        * above
    )

    return torch.mean(
        widths
        + penalty_below
        + penalty_above
    ).item()


def metric_rho_risk(
    y_true,
    y_pred,
    quantiles=(0.1, 0.5, 0.9),
    eps=1e-8
):

    y_true, y_pred = _sync_tensors(
        y_true,
        y_pred
    )

    total_true = torch.clamp(
        torch.sum(
            torch.abs(y_true)
        ),
        min=eps
    )

    values = []

    for q in quantiles:

        yq = torch.quantile(
            y_pred,
            q,
            dim=-1
        )

        error = (
            y_true - yq
        )

        loss = torch.maximum(
            q * error,
            (q - 1) * error
        )

        values.append(
            2 * torch.sum(loss)
            / total_true
        )

    return torch.mean(
        torch.stack(values)
    ).item()

# --- CELL ---

# ============================================================
# CELL 15 — MODEL FACTORY
# ============================================================

def make_model(
    model_name,
    likelihood=None
):

    common = dict(
        input_chunk_length=24,
        output_chunk_length=PRED_LEN,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer_kwargs={
            "lr": LEARNING_RATE
        },
        random_state=SEED,
        force_reset=True,
    )

    if likelihood is not None:
        common["likelihood"] = likelihood

    if model_name == "NBEATS":
        return NBEATSModel(
            **common
        )

    elif model_name == "NHiTS":
        return NHiTSModel(
            **common
        )

    elif model_name == "TCN":
        return TCNModel(
            input_chunk_length=30,
            output_chunk_length=PRED_LEN,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            optimizer_kwargs={
                "lr": LEARNING_RATE
            },
            random_state=SEED,
            likelihood=likelihood,
        )

    elif model_name == "Transformer":
        return TransformerModel(
            **common
        )

    elif model_name == "RNN":
        return RNNModel(
            model="LSTM",
            **common
        )

    elif model_name == "BlockRNN":
        return BlockRNNModel(
            model="LSTM",
            **common
        )

    elif model_name == "DLinear":
        return DLinearModel(
            **common
        )

    elif model_name == "NLinear":
        return NLinearModel(
            normalize=False,
            **common
        )

    elif model_name == "TiDE":
        return TiDEModel(
            **common
        )

    elif model_name == "TSMixer":
        return TSMixerModel(
            **common
        )

    elif model_name == "TFT":

        return TFTModel(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            optimizer_kwargs={
                "lr": LEARNING_RATE
            },
            random_state=SEED,
            likelihood=likelihood,
            add_relative_index=True,
        )

    elif model_name == "Chronos2":
        return Chronos2Model(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            random_state=SEED,
            n_epochs=N_EPOCHS,
        )

    elif model_name == "TimesFM2p5":
        return TimesFM2p5Model(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            random_state=SEED,
            n_epochs=N_EPOCHS,
        )


    elif model_name == "PatchTSTFM":
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("patchtst", "../../models/darts-original/patchtst_fm_model.py")
        patchtst = importlib.util.module_from_spec(spec)
        sys.modules["patchtst"] = patchtst
        spec.loader.exec_module(patchtst)
        return patchtst.PatchTSTFMModel(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            n_epochs=N_EPOCHS,
        )

    elif model_name == "TiREx":
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("tirex", "../../models/darts-original/tirex_model.py")
        tirex = importlib.util.module_from_spec(spec)
        sys.modules["tirex"] = tirex
        spec.loader.exec_module(tirex)
        return tirex.TiRExModel(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            n_epochs=N_EPOCHS,
            accept_license=True,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

# --- CELL ---

# ============================================================
# CELL 16 — MODELS
# ============================================================

MODELS = [
    "BlockRNN",    # Tests standard DL + both Gaussian/Quantile likelihoods
    "Chronos2",    # Tests native zero-shot Foundation Models
    "PatchTSTFM"   # Tests your custom dynamic-import models
]


print("Models:")
for x in MODELS:
    print(" -", x)

# --- CELL ---

# ============================================================
# CELL 17 - COVARIATE SUPPORT
# ============================================================

def get_fit_kwargs(
    model,
    target,
    past_covs=None
):

    kwargs = {
        "series": target,
        "verbose": True,
        "dataloader_kwargs": {"num_workers": 0},
    }

    if getattr(
        model,
        "supports_past_covariates",
        False
    ):
        if past_covs is not None:
            kwargs["past_covariates"] = past_covs
        else:
            kwargs["past_covariates"] = base_train_pc

    return kwargs


# --- CELL ---

# ============================================================
# CELL 18 — TRAIN / EVALUATE LIKELIHOOD MODEL
# ============================================================

def evaluate_base_model(
    model,
    test_windows,
    likelihood_name,
    model_name,
):

    window_metrics = []

    total_inference_time = 0.0

    for window_id, window_df in enumerate(
        test_windows
    ):

        print(
            f"      Test window "
            f"{window_id + 1}/{len(test_windows)}"
        )

        full_ts = (
            TimeSeries
            .from_dataframe(window_df)
            .astype(np.float32)
        )

        full_sc = (
            y_scaler
            .transform(full_ts)
            .astype(np.float32)
        )

        full_pc = (
            build_past_covs(
                full_sc,
                lags=LAGS
            )
            .astype(np.float32)
        )

        full_sc = (
            full_sc
            .slice_intersect(full_pc)
        )

        full_ts = (
            full_ts
            .slice_intersect(full_sc)
        )

        full_pc = (
            full_pc
            .slice_intersect(full_sc)
        )

        past_sc = full_sc[:-PRED_LEN]

        past_original = (
            full_ts[:-PRED_LEN]
        )

        gt_future = (
            full_ts[-PRED_LEN:]
        )

        forecast_start = (
            gt_future.start_time()
        )

        pc_past = (
            full_pc
            .drop_after(forecast_start, keep_point=False)
        )

        predict_kwargs = {}

        if getattr(
            model,
            "supports_past_covariates",
            False
        ):
            predict_kwargs[
                "past_covariates"
            ] = pc_past

        start = time.time()

        fc_sc = model.predict(
            n=PRED_LEN,
            series=past_sc,
            num_samples=NUM_PRED_SAMPLES,
            random_state=SEED,
            verbose=False,
            **predict_kwargs,
        )

        inference_time = (
            time.time() - start
        )

        total_inference_time += (
            inference_time
        )

        fc = y_scaler.inverse_transform(
            fc_sc
        )

        fc = fc.with_values(np.clip(fc.all_values(), a_min=0, a_max=None))

        y_pred = fc.all_values(
            copy=False
        )

        y_true = (
            gt_future
            .all_values(copy=False)
            .squeeze(-1)
        )

        y_train = (
            past_original
            .all_values(copy=False)
            .squeeze(-1)
        )

        metrics = {

            "MAE":
                metric_mae(
                    y_true,
                    y_pred
                ),

            "MSE":
                metric_mse(
                    y_true,
                    y_pred
                ),

            "RMSE":
                metric_rmse(
                    y_true,
                    y_pred
                ),

            "MAPE":
                metric_mape(
                    y_true,
                    y_pred
                ),

            "sMAPE":
                metric_smape(
                    y_true,
                    y_pred
                ),

            "MASE":
                metric_mase(
                    y_true,
                    y_pred,
                    y_train
                ),

            "RMSSE":
                metric_rmsse(
                    y_true,
                    y_pred,
                    y_train
                ),

            "CRPS":
                metric_crps(
                    y_true,
                    y_pred
                ),

            "PICP":
                metric_picp(
                    y_true,
                    y_pred,
                    alpha=ALPHA
                ),

            "MIS":
                metric_mis(
                    y_true,
                    y_pred,
                    alpha=ALPHA
                ),

            "MPIW":
                metric_mpiw(
                    y_pred,
                    alpha=ALPHA
                ),

            "Rho_Risk":
                metric_rho_risk(
                    y_true,
                    y_pred
                ),
        }

        window_metrics.append(
            metrics
        )

    summary = pd.DataFrame(
        window_metrics
    ).mean()

    summary[
        "Inference Time (s)"
    ] = (
        total_inference_time
        / len(test_windows)
    )

    return summary

# --- CELL ---

from darts.utils.likelihood_models import (
    GaussianLikelihood,
    QuantileRegression
)

LIKELIHOODS = {
    "Gaussian": GaussianLikelihood(),

    "Quantile": QuantileRegression(
        quantiles=QUANTILES
    ),

}

# --- CELL ---

# ============================================================
# CELL 19 — RUN ALL LIKELIHOOD BASELINES
# ============================================================

results = []

CSV_FILE = (
    "Solar_Darts_Likelihood_Baselines.csv"
)

for model_name in MODELS:

    for likelihood_name in LIKELIHOODS:
        # Foundation models only support Quantile in this setup
        if model_name in ["Chronos2", "TimesFM2p5", "PatchTSTFM", "TiREx"] and likelihood_name != "Quantile":
            print(f"Skipping {likelihood_name} for {model_name} (Zero-shot Foundation Models default to Quantile)")
            continue


        print("\n")
        print("=" * 70)
        print(
            f"MODEL: {model_name} | "
            f"LIKELIHOOD: {likelihood_name}"
        )
        print("=" * 70)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:

            likelihood = (
                LIKELIHOODS[
                    likelihood_name
                ]
            )

            model = make_model(
                model_name,
                likelihood=likelihood
            )

            print("Training...")

            start_train = time.time()

            fit_kwargs = (
                get_fit_kwargs(
                    model,
                    train_y_sc,
                    past_covs=train_pc
                )
            )

            model.fit(
                **fit_kwargs
            )

            training_time = (
                time.time()
                - start_train
            )

            print(
                f"Training time: "
                f"{training_time:.2f}s"
            )

            print("Evaluating...")

            metrics = (
                evaluate_base_model(
                    model=model,
                    test_windows=test_windows,
                    likelihood_name=likelihood_name,
                    model_name=model_name,
                )
            )

            row = {
                "Model": model_name,
                "Method": likelihood_name,
                "N_Epochs": N_EPOCHS,
                "Batch Size": BATCH_SIZE,
                "Prediction Samples":
                    NUM_PRED_SAMPLES,
                "Training Time (s)":
                    training_time,
            }

            for metric_name, value in (
                metrics.items()
            ):
                row[metric_name] = value

            results.append(row)

            pd.DataFrame(
                results
            ).to_csv(
                CSV_FILE,
                index=False
            )

            print(
                pd.DataFrame(
                    [row]
                ).T
            )

        except Exception as e:

            print(
                f"FAILED: "
                f"{model_name} / "
                f"{likelihood_name}"
            )

            print(
                type(e).__name__,
                str(e)
            )

            continue

print("\nFinished likelihood benchmark.")

likelihood_results = pd.DataFrame(
    results
)

likelihood_results

# --- CELL ---

# ============================================================
# CELL 20 — CONFORMAL CALIBRATION HELPER
# ============================================================

def calibrate_conformal_model(
    base_model,
    conformal_type="naive"
):

    if conformal_type == "naive":

        cp_model = ConformalNaiveModel(
            model=base_model,
            quantiles=QUANTILES,
            symmetric=True,
            cal_length=CAL_LENGTH,
            cal_stride=1,
            cal_num_samples=NUM_PRED_SAMPLES,
            random_state=SEED,
        )

    elif conformal_type == "qr":

        cp_model = ConformalQRModel(
            model=base_model,
            quantiles=QUANTILES,
            symmetric=True,
            cal_length=CAL_LENGTH,
            cal_stride=1,
            cal_num_samples=NUM_PRED_SAMPLES,
            random_state=SEED,
        )

    else:

        raise ValueError(
            "Unknown conformal type"
        )

    return cp_model

# --- CELL ---

# ============================================================
# CELL 21 — CONFORMAL NAIVE BASE MODEL FACTORY
# ============================================================

def make_conformal_naive_base(
    model_name
):

    # Gaussian base model
    likelihood = GaussianLikelihood()

    return make_model(
        model_name,
        likelihood=likelihood
    )

# --- CELL ---

# ============================================================
# CELL 22 — EVALUATE CONFORMAL MODEL
# ============================================================

def evaluate_conformal_model(
    cp_model,
    test_windows,
    conformal_name,
):

    window_metrics = []

    total_inference_time = 0.0

    for window_id, window_df in enumerate(
        test_windows
    ):

        print(
            f"      Test window "
            f"{window_id + 1}/{len(test_windows)}"
        )

        full_ts = (
            TimeSeries
            .from_dataframe(window_df)
            .astype(np.float32)
        )

        full_sc = (
            y_scaler
            .transform(full_ts)
            .astype(np.float32)
        )

        full_pc = (
            build_past_covs(
                full_sc,
                lags=LAGS
            )
            .astype(np.float32)
        )

        full_sc = (
            full_sc
            .slice_intersect(full_pc)
        )

        full_ts = (
            full_ts
            .slice_intersect(full_sc)
        )

        full_pc = (
            full_pc
            .slice_intersect(full_sc)
        )

        past_sc = full_sc[:-PRED_LEN]

        past_original = (
            full_ts[:-PRED_LEN]
        )

        gt_future = (
            full_ts[-PRED_LEN:]
        )

        forecast_start = (
            gt_future.start_time()
        )

        pc_past = (
            full_pc
            .drop_after(forecast_start, keep_point=False)
        )

        predict_kwargs = {}

        if getattr(
            cp_model,
            "supports_past_covariates",
            False
        ):
            predict_kwargs[
                "past_covariates"
            ] = pc_past

        start = time.time()

        fc_sc = cp_model.predict(
            n=PRED_LEN,
            series=past_sc,
            num_samples=NUM_PRED_SAMPLES,
            random_state=SEED,
            verbose=False,
            **predict_kwargs,
        )

        inference_time = (
            time.time() - start
        )

        total_inference_time += (
            inference_time
        )

        fc = y_scaler.inverse_transform(
            fc_sc
        )

        fc = fc.with_values(np.clip(fc.all_values(), a_min=0, a_max=None))

        y_pred = fc.all_values(
            copy=False
        )

        y_true = (
            gt_future
            .all_values(copy=False)
            .squeeze(-1)
        )

        y_train = (
            past_original
            .all_values(copy=False)
            .squeeze(-1)
        )

        metrics = {

            "MAE":
                metric_mae(
                    y_true,
                    y_pred
                ),

            "MSE":
                metric_mse(
                    y_true,
                    y_pred
                ),

            "RMSE":
                metric_rmse(
                    y_true,
                    y_pred
                ),

            "MAPE":
                metric_mape(
                    y_true,
                    y_pred
                ),

            "sMAPE":
                metric_smape(
                    y_true,
                    y_pred
                ),

            "MASE":
                metric_mase(
                    y_true,
                    y_pred,
                    y_train
                ),

            "RMSSE":
                metric_rmsse(
                    y_true,
                    y_pred,
                    y_train
                ),

            "CRPS":
                metric_crps(
                    y_true,
                    y_pred
                ),

            "PICP":
                metric_picp(
                    y_true,
                    y_pred,
                    alpha=ALPHA
                ),

            "MIS":
                metric_mis(
                    y_true,
                    y_pred,
                    alpha=ALPHA
                ),

            "MPIW":
                metric_mpiw(
                    y_pred,
                    alpha=ALPHA
                ),

            "Rho_Risk":
                metric_rho_risk(
                    y_true,
                    y_pred
                ),
        }

        window_metrics.append(
            metrics
        )

    summary = pd.DataFrame(
        window_metrics
    ).mean()

    summary[
        "Inference Time (s)"
    ] = (
        total_inference_time
        / len(test_windows)
    )

    return summary

# --- CELL ---

# ============================================================
# CELL 23 — CONFORMAL NAIVE
# ============================================================

conformal_results = []

CONFORMAL_CSV = (
    "Solar_Darts_Conformal_Baselines.csv"
)

for model_name in MODELS:

    print("\n")
    print("=" * 70)
    print(
        f"CONFORMAL NAIVE | {model_name}"
    )
    print("=" * 70)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:

        # ----------------------------------------------------
        # 1. Train base model ONLY on base training region
        # ----------------------------------------------------

        base_model = (
            make_conformal_naive_base(
                model_name
            )
        )

        print(
            "Training base model..."
        )

        start_train = time.time()

        fit_kwargs = (
            get_fit_kwargs(
                base_model,
                base_train_y_sc
            )
        )

        base_model.fit(
            **fit_kwargs
        )

        training_time = (
            time.time()
            - start_train
        )

        print(
            f"Base training time: "
            f"{training_time:.2f}s"
        )

        # ----------------------------------------------------
        # 2. Create conformal wrapper
        # ----------------------------------------------------

        cp_model = (
            calibrate_conformal_model(
                base_model,
                conformal_type="naive"
            )
        )

        # ----------------------------------------------------
        # 3. Calibrate using HELD-OUT calibration region
        # ----------------------------------------------------

        print(
            "Calibrating..."
        )

        calibration_kwargs = {}

        if getattr(
            cp_model,
            "supports_past_covariates",
            False
        ):
            calibration_kwargs[
                "past_covariates"
            ] = calibration_pc

        # Darts conformal model uses the
        # calibration series supplied to predict().
        #
        # We perform one calibration pass by asking
        # it to generate forecasts over the calibration
        # region.

        cp_model.fit(
            calibration_y_sc,
            **calibration_kwargs
        )

        # ----------------------------------------------------
        # 4. Evaluate
        # ----------------------------------------------------

        metrics = (
            evaluate_conformal_model(
                cp_model,
                test_windows,
                "ConformalNaive"
            )
        )

        row = {
            "Model": model_name,
            "Method": "ConformalNaive",
            "N_Epochs": N_EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Calibration Length": CAL_LENGTH,
            "Prediction Samples":
                NUM_PRED_SAMPLES,
            "Training Time (s)":
                training_time,
        }

        for metric_name, value in (
            metrics.items()
        ):
            row[metric_name] = value

        conformal_results.append(row)

        pd.DataFrame(
            conformal_results
        ).to_csv(
            CONFORMAL_CSV,
            index=False
        )

        print(
            pd.DataFrame(
                [row]
            ).T
        )

    except Exception as e:

        print(
            f"FAILED: "
            f"{model_name} / "
            f"ConformalNaive"
        )

        print(
            type(e).__name__,
            str(e)
        )

        continue

# --- CELL ---

# ============================================================
# CELL 24 — CHECK DARTS VERSION
# ============================================================

import darts

print(
    "Darts version:",
    darts.__version__
)

print(
    "ConformalNaiveModel:",
    ConformalNaiveModel
)

print(
    "ConformalQRModel:",
    ConformalQRModel
)

# --- CELL ---

# ============================================================
# CELL 25 — CONFORMAL QR BASE MODEL
# ============================================================

def make_conformal_qr_base(
    model_name
):

    likelihood = QuantileRegression(
        quantiles=QUANTILES
    )

    return make_model(
        model_name,
        likelihood=likelihood
    )

# --- CELL ---

# ============================================================
# CELL 26 — CONFORMAL QR
# ============================================================

for model_name in MODELS:

    print("\n")
    print("=" * 70)
    print(
        f"CONFORMAL QR | {model_name}"
    )
    print("=" * 70)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:

        # ----------------------------------------------------
        # 1. Train probabilistic quantile base model
        # ----------------------------------------------------

        base_model = (
            make_conformal_qr_base(
                model_name
            )
        )

        print(
            "Training quantile base model..."
        )

        start_train = time.time()

        fit_kwargs = (
            get_fit_kwargs(
                base_model,
                base_train_y_sc
            )
        )

        base_model.fit(
            **fit_kwargs
        )

        training_time = (
            time.time()
            - start_train
        )

        print(
            f"Training time: "
            f"{training_time:.2f}s"
        )

        # ----------------------------------------------------
        # 2. Conformal QR wrapper
        # ----------------------------------------------------

        cp_model = (
            ConformalQRModel(
                model=base_model,
                quantiles=QUANTILES,
                symmetric=True,
                cal_length=CAL_LENGTH,
                cal_stride=1,
                cal_num_samples=NUM_PRED_SAMPLES,
                random_state=SEED,
            )
        )

        # ----------------------------------------------------
        # 3. Calibration
        # ----------------------------------------------------

        print(
            "Calibrating conformal QR..."
        )

        calibration_kwargs = {}

        if getattr(
            cp_model,
            "supports_past_covariates",
            False
        ):
            calibration_kwargs[
                "past_covariates"
            ] = calibration_pc

        cp_model.fit(
            calibration_y_sc,
            **calibration_kwargs
        )

        # ----------------------------------------------------
        # 4. Test evaluation
        # ----------------------------------------------------

        metrics = (
            evaluate_conformal_model(
                cp_model,
                test_windows,
                "ConformalQR"
            )
        )

        row = {
            "Model": model_name,
            "Method": "ConformalQR",
            "N_Epochs": N_EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Calibration Length": CAL_LENGTH,
            "Prediction Samples":
                NUM_PRED_SAMPLES,
            "Training Time (s)":
                training_time,
        }

        for metric_name, value in (
            metrics.items()
        ):
            row[metric_name] = value

        conformal_results.append(row)

        pd.DataFrame(
            conformal_results
        ).to_csv(
            CONFORMAL_CSV,
            index=False
        )

        print(
            pd.DataFrame(
                [row]
            ).T
        )

    except Exception as e:

        print(
            f"FAILED: "
            f"{model_name} / "
            f"ConformalQR"
        )

        print(
            type(e).__name__,
            str(e)
        )

        continue

# --- CELL ---

# ============================================================
# CELL 27 — COMBINE RESULTS
# ============================================================

all_baseline_results = pd.concat(
    [
        likelihood_results,
        pd.DataFrame(
            conformal_results
        )
    ],
    ignore_index=True,
    sort=False
)

FINAL_CSV = (
    "Solar_Darts_All_Baselines.csv"
)

all_baseline_results.to_csv(
    FINAL_CSV,
    index=False
)

print(
    "Saved:",
    FINAL_CSV
)

all_baseline_results

# --- CELL ---

# ============================================================
# CELL 28 — RANK BY CRPS
# ============================================================

ranking = (
    all_baseline_results
    .sort_values(
        "CRPS",
        ascending=True
    )
    .reset_index(drop=True)
)

columns = [
    "Model",
    "Method",
    "MAE",
    "MASE",
    "RMSSE",
    "CRPS",
    "PICP",
    "MIS",
    "MPIW",
    "Rho_Risk",
    "Training Time (s)",
    "Inference Time (s)",
]

ranking[
    [
        c for c in columns
        if c in ranking.columns
    ]
]

# --- CELL ---

# ============================================================
# CELL 29 — COVERAGE ANALYSIS
# ============================================================

coverage = ranking[
    [
        "Model",
        "Method",
        "PICP",
        "MPIW",
        "MIS",
        "CRPS",
    ]
].copy()

coverage[
    "Coverage Error"
] = np.abs(
    coverage["PICP"] - 0.95
)

coverage = coverage.sort_values(
    "Coverage Error"
)

coverage# ============================================================
# CELL 30 — FINAL CLEAN TABLE
# ============================================================

final_table = ranking[
    [
        "Model",
        "Method",
        "MAE",
        "MASE",
        "RMSSE",
        "CRPS",
        "PICP",
        "MIS",
        "MPIW",
        "Rho_Risk",
    ]
].copy()

final_table

# --- CELL ---

# ============================================================
# CELL 31 — SAVE FINAL OUTPUTS
# ============================================================

all_baseline_results.to_csv(
    "/kaggle/working/Solar_Darts_All_Baselines.csv",
    index=False
)

ranking.to_csv(
    "/kaggle/working/Solar_Darts_Ranked.csv",
    index=False
)

coverage.to_csv(
    "/kaggle/working/Solar_Darts_Coverage.csv",
    index=False
)

print("Files saved:")
print(
    "/kaggle/working/Solar_Darts_All_Baselines.csv"
)
print(
    "/kaggle/working/Solar_Darts_Ranked.csv"
)
print(
    "/kaggle/working/Solar_Darts_Coverage.csv"
)