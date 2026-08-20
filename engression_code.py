import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import DataLoader

---CELL_BOUNDARY---

%%capture
!pip install darts
!pip install gluonts
!pip install lightning
!pip install neuralforecast
# Optional: install tirex dependency for EnTiRexModel
!pip install tirex-ts


---CELL_BOUNDARY---

import os
import random
import numpy as np
import torch
import cv2
import pytorch_lightning as pl

from transformers import set_seed
from datasets import disable_progress_bar
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Deterministic:
    def _init_(self):
        pass

    def init_all(self, seed=0, disable_list=['cuda_block']):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if 'cuda_block' not in disable_list: 
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if 'torch_deter_algo' not in disable_list: # consumn more gpu sometimes
            torch.use_deterministic_algorithms(True, warn_only=True)
        set_seed(seed)
        cv2.setRNGSeed(seed)
        disable_progress_bar()

deterministic = Deterministic()

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

---CELL_BOUNDARY---

import os
import sys
import shutil
import subprocess

repo_path = "timeseries"
if not os.path.exists(repo_path):
    subprocess.run(["git", "clone", "https://github.com/pymaxx13/timeseries.git"])

# Add the repository to Python's import path
repo_full_path = os.path.abspath(repo_path)
if repo_full_path not in sys.path:
    sys.path.insert(0, repo_full_path)

print("Repository exists:", os.path.exists(repo_full_path))
print(
    "engressionts exists:",
    os.path.exists(os.path.join(repo_full_path, "engressionts"))
)
print("Repo path:", repo_full_path)
print("Python path contains repo:", repo_full_path in sys.path)


---CELL_BOUNDARY---

import os
print(os.getcwd())

---CELL_BOUNDARY---

import sys
from pathlib import Path

project_root = Path.cwd().parents[2]   # .../engression-ts
sys.path.insert(0, str(project_root))

---CELL_BOUNDARY---

# import traceback

# print("=" * 80)
# print("Testing EnBEATS import...")
# print("=" * 80)

# try:
#     from engressionts.models.darts.nbeats import EnBEATSModel
#     print("\n✅ SUCCESS")
#     print(EnBEATSModel)

# except Exception as e:
#     print("\n❌ IMPORT FAILED")
#     print(f"\nException type: {type(e).__name__}")
#     print(f"\nException message:\n{e}")

#     print("\nFull traceback:")
#     traceback.print_exc()

# print("\n" + "=" * 80)

---CELL_BOUNDARY---

# import traceback

# print("=" * 80)
# print("Testing EnHiTS import...")
# print("=" * 80)

# try:
#     from engressionts.models.darts.nhits import EnHiTSModel
#     print("\n✅ SUCCESS")
#     print(EnHiTSModel)

# except Exception as e:
#     print("\n❌ IMPORT FAILED")
#     print(f"\nException type: {type(e).__name__}")
#     print(f"\nException message:\n{e}")

#     print("\nFull traceback:")
#     traceback.print_exc()

# print("\n" + "=" * 80)

---CELL_BOUNDARY---

import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple, Optional

from darts import TimeSeries
from darts.metrics import mae, rmse, smape, mase, rmsse
from darts.dataprocessing.transformers import Scaler

# Change this import to the model you want to evaluate.
# Example: EnBEATSModel, EnTransformerModel, EnTCNModel, etc.
from engressionts.models.darts import EnHiTSModel, EnTransformerModel, EnTCNModel, EnBlockRNNModel, EnDLinearModel, EnBEATSModel

---CELL_BOUNDARY---

PRED_LEN = 24
LAGS = (1, 24, 168)
DATASET_NAME = "solar_nips"

---CELL_BOUNDARY---

import numpy as np
import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from darts import TimeSeries

def gluonts_item_to_darts_mv(item, freq: str) -> TimeSeries:
    # GluonTS start is often a pandas Period; convert safely
    start = item["start"].to_timestamp() if hasattr(item["start"], "to_timestamp") else pd.Timestamp(item["start"])

    # GluonTS multivariate target is typically shape (D, T)
    target = np.asarray(item["target"])
    if target.ndim != 2:
        raise ValueError(f"Expected multivariate target with ndim=2, got shape {target.shape}")

    values = target.T  # Darts expects (T, D)
    times = pd.date_range(start=start, periods=values.shape[0], freq=freq)
    cols = [f"dim_{i}" for i in range(values.shape[1])]

    return TimeSeries.from_times_and_values(times, values, columns=cols)

ds = get_dataset(DATASET_NAME, regenerate=False)
freq = ds.metadata.freq

# Determine number of rolling test windows
train_list = list(ds.train)
test_list  = list(ds.test)
num_test_dates = len(test_list) // len(train_list)

# Automatically determines the number of target variables from the dataset.
target_dim = int(ds.metadata.feat_static_cat[0].cardinality)

train_grouper = MultivariateGrouper(max_target_dim=target_dim)
test_grouper  = MultivariateGrouper(num_test_dates=num_test_dates, max_target_dim=target_dim)

train_mv_items = list(train_grouper(train_list))
test_mv_items  = list(test_grouper(test_list))

train_ts = gluonts_item_to_darts_mv(train_mv_items[0], freq)
test_ts_list = [gluonts_item_to_darts_mv(it, freq) for it in test_mv_items]

---CELL_BOUNDARY---

from darts.dataprocessing.transformers import Scaler

y_scaler = Scaler()                     # StandardScaler-like wrapper
train_y_sc = y_scaler.fit_transform(train_ts)

import numpy as np
from darts import TimeSeries, concatenate

# Change the lag values if you want to use a different lag configuration.
def lag_covs_from_scaled_target(ts_sc: TimeSeries, lags=(1,24,168)) -> TimeSeries:
    shifted = []
    for L in lags:
        s = ts_sc.shift(L).with_columns_renamed(
            ts_sc.components, [f"{c}_lag{L}" for c in ts_sc.components]
        )
        shifted.append(s)

    common = shifted[0]
    for s in shifted[1:]:
        common = common.slice_intersect(s)
    shifted = [s.slice_intersect(common) for s in shifted]
    return concatenate(shifted, axis=1)

def fourier_from_index(idx) -> TimeSeries:
    hour = idx.hour.to_numpy()
    dow  = idx.dayofweek.to_numpy()
    X = np.vstack([
        np.sin(2*np.pi*hour/24.0),
        np.cos(2*np.pi*hour/24.0),
        np.sin(2*np.pi*dow/7.0),
        np.cos(2*np.pi*dow/7.0),
    ]).T
    return TimeSeries.from_times_and_values(idx, X, columns=["h_sin","h_cos","dow_sin","dow_cos"])

def dim_indicator_norm(idx, D: int) -> TimeSeries:
    v = (np.arange(D, dtype=np.float32) / (D-1)).astype(np.float32)  # 0..1
    X = np.tile(v, (len(idx), 1))
    cols = [f"dim_id_{i}" for i in range(D)]
    return TimeSeries.from_times_and_values(idx, X, columns=cols)

# Builds past covariates using lag features, node IDs, and time features.
def build_past_covs_552(ts_sc: TimeSeries, lags=(1,24,168)) -> TimeSeries:
    lag_covs = lag_covs_from_scaled_target(ts_sc, lags)
    idx = lag_covs.time_index
    time_covs = fourier_from_index(idx)
    dim_covs  = dim_indicator_norm(idx, ts_sc.width)
    return concatenate([lag_covs, dim_covs, time_covs], axis=1)


def ts_upto(ts, end_time):
    # Compatible with multiple Darts versions.
    if hasattr(ts, "slice_end"):
        return ts.slice_end(end_time)
    if hasattr(ts, "drop_after"):
        return ts.drop_after(end_time)
    if hasattr(ts, "split_after"):
        return ts.split_after(end_time)[0]
    return ts.slice(ts.start_time(), end_time)

---CELL_BOUNDARY---

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset

def gluon_to_wide_df(dataset):
    series_list = []
    
    for i, entry in enumerate(dataset):
        # 1. Create the time index using the start date and frequency
        idx = pd.date_range(
            start=entry["start"].to_timestamp(), 
            periods=len(entry["target"]), 
            freq=entry["start"].freqstr
        )
        
        # 2. Create a Series for each node, named by its index (0 to 136)
        series = pd.Series(entry["target"], index=idx, name=f"node_{i}")
        series_list.append(series)
    
    # 3. Concatenate all series into one DataFrame (T x N)
    return pd.concat(series_list, axis=1)

# Usage:
dataset = get_dataset(DATASET_NAME, regenerate=False)
df_train = gluon_to_wide_df(dataset.train)

def get_test_windows(dataset, num_nodes=137):
    all_series = []
    
    # 1. Convert everything to a list of Series first
    for entry in dataset:
        idx = pd.date_range(
            start=entry["start"].to_timestamp(), 
            periods=len(entry["target"]), 
            freq=entry["start"].freqstr
        )
        # Give them a generic name for now
        all_series.append(pd.Series(entry["target"], index=idx))
    
    # 2. Split the list into 7 chunks of 137
    # This assumes the order is [Node0_W1, Node1_W1... Node136_W1, Node0_W2...]
    num_windows = len(all_series) // num_nodes
    windows = []
    
    for w in range(num_windows):
        start_idx = w * num_nodes
        end_idx = (w + 1) * num_nodes
        
        # Grab 137 series and concat them side-by-side
        window_df = pd.concat(all_series[start_idx:end_idx], axis=1)
        
        # Rename columns to node_0, node_1... node_136
        window_df.columns = [f"node_{i}" for i in range(num_nodes)]
        windows.append(window_df)
        
    return windows

# Execute
test_windows = get_test_windows(dataset.test, num_nodes=137)

---CELL_BOUNDARY---

df_train.shape

---CELL_BOUNDARY---

def crps(preds, targets, quantiles=(np.arange(20) / 20.0)[1:]):
    """
    preds: (B, N, T, D) or (B, N, T)
    targets: (B, T, D) or (B, T)
    """
    x = np.quantile(preds, quantiles, axis=1, method="nearest")  # -> (Q, B, T, D) or (Q, B, T)
    quantiles = np.expand_dims(quantiles, axis=list(range(1, len(preds.shape))))  # (Q,1,1,1)
    loss = 2 * np.sum(np.abs((x - targets) * ((targets <= x) - quantiles)), axis=2)  # sum over T
    return loss.mean() / np.abs(targets).sum(axis=1).mean()


def crps_sum_like_theirs(preds, targets, quantiles=(np.arange(20) / 20.0)[1:], frequency='D'):
    # preds: (B, N, T, D)
    # targets: (B, T, D)

    preds_sum = preds.sum(axis=-1)      # (B, N, T)
    targets_sum = targets.sum(axis=-1)  # (B, T)

    return crps(preds_sum, targets_sum, quantiles=quantiles)


def get_crps(model, test_windows, y_scaler, pred_len=24, lags=(1, 24, 168), num_samples=100, seed=42, std=None):
    all_forecasts = []
    all_targets = []

    for i, window_df in enumerate(test_windows):
        full_ts = TimeSeries.from_dataframe(window_df).astype(np.float32)
        full_sc = y_scaler.transform(full_ts).astype(np.float32)
        full_pc = build_past_covs_552(full_sc, lags=lags).astype(np.float32)

        full_sc = full_sc.slice_intersect(full_pc)
        full_ts = full_ts.slice_intersect(full_sc)
        full_pc = full_pc.slice_intersect(full_sc)

        past_true_sc = full_sc[:-pred_len]
        gt_future = full_ts[-pred_len:]

        forecast_start = gt_future.start_time()
        pc_past = ts_upto(full_pc, forecast_start)

        fc_sc = model.predict(
            n=pred_len,
            series=past_true_sc,
            past_covariates=pc_past,
            num_samples=num_samples,
            verbose=False,
            random_state=seed,
        )

        fc = y_scaler.inverse_transform(fc_sc)
        fc = fc.with_values(np.clip(fc.all_values(), a_min=0, a_max=None))

        assert fc.time_index.equals(gt_future.time_index), \
            f"Forecast index mismatch in window {i}"

        all_forecasts.append(fc.all_values(copy=False))
        all_targets.append(gt_future.all_values(copy=False))

    stacked_forecasts = np.stack(all_forecasts, axis=0)      # (B, T, D, N)
    preds_reshaped = np.transpose(stacked_forecasts, (0, 3, 1, 2))  # (B, N, T, D)

    stacked_targets = np.stack(all_targets, axis=0)          # (B, T, D, 1)
    targets_reshaped = np.squeeze(stacked_targets, axis=-1)  # (B, T, D)

    return crps_sum_like_theirs(preds_reshaped, targets_reshaped)

---CELL_BOUNDARY---

import torch
import numpy as np

# Utility functions
def _sync_tensors(*args):
    """
    Converts inputs to PyTorch tensors and synchronizes them to the same device.
    Uses the device of the first tensor found.
    """
    tensors = [torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x for x in args]
    if not tensors:
        return []
    
    device = tensors[0].device
    return [t.to(device, dtype=torch.float32) for t in tensors]

def _get_point_forecast(y_pred, point_method):
    """
    Collapses the sample dimension (dim=-1) to generate a point forecast.
    """
    if point_method == "mean":
        return torch.mean(y_pred, dim=-1)
    elif point_method == "median":
        return torch.median(y_pred, dim=-1).values
    elif isinstance(point_method, float) and 0.0 <= point_method <= 1.0:
        return torch.quantile(y_pred, point_method, dim=-1)
    else:
        raise ValueError("point_method must be 'mean', 'median', or a float between 0.0 and 1.0")

def _aggregate(tensor, method):
    """
    Aggregates the tensor over the remaining dimensions.
    """
    if method is None or method == "none":
        return tensor
    elif method == "mean":
        return torch.mean(tensor)
    elif method == "median":
        return torch.median(tensor)
    elif method == "sum":
        return torch.sum(tensor)
    elif method == "norm":
        return torch.norm(tensor)
    else:
        raise ValueError(f"Unsupported aggregate_method: {method}")

def _format_return(tensor):
    return tensor.item() if tensor.numel() == 1 else tensor


# Point metrics
def mae(y_true, y_pred, point_method="mean", aggregate_method="mean"):
    """
    Mean Absolute Error (MAE).
    
    .. math::
        \\text{MAE} = \\frac{1}{N} \\sum_{i=1}^{N} |y_i - \\hat{y}_i|
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_point = _get_point_forecast(y_pred, point_method)
    error = torch.abs(y_true - y_point)
    return _format_return(_aggregate(error, aggregate_method))

def mse(y_true, y_pred, point_method="mean", aggregate_method="mean"):
    """
    Mean Squared Error (MSE).
    
    .. math::
        \\text{MSE} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_point = _get_point_forecast(y_pred, point_method)
    error = (y_true - y_point) ** 2
    return _format_return(_aggregate(error, aggregate_method))

def rmse(y_true, y_pred, point_method="mean", aggregate_method="mean"):
    """
    Root Mean Squared Error (RMSE).
    
    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2}
    """
    mse_val = mse(y_true, y_pred, point_method, aggregate_method="none")
    return _format_return(_aggregate(torch.sqrt(mse_val), aggregate_method))

def mape(y_true, y_pred, point_method="mean", aggregate_method="mean", eps=1e-8):
    """
    Mean Absolute Percentage Error (MAPE).
    
    .. math::
        \\text{MAPE} = \\frac{100}{N} \\sum_{i=1}^{N} \\left| \\frac{y_i - \\hat{y}_i}{\\max(|y_i|, \\epsilon)} \\right|
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_point = _get_point_forecast(y_pred, point_method)
    error = torch.abs((y_true - y_point) / torch.clamp(torch.abs(y_true), min=eps))
    return _format_return(_aggregate(error * 100, aggregate_method))

def smape(y_true, y_pred, point_method="mean", aggregate_method="mean", eps=1e-8):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    
    .. math::
        \\text{sMAPE} = \\frac{100}{N} \\sum_{i=1}^{N} \\frac{|y_i - \\hat{y}_i|}{\\max((|y_i| + |\\hat{y}_i|) / 2, \\epsilon)}
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_point = _get_point_forecast(y_pred, point_method)
    num = torch.abs(y_point - y_true)
    denom = torch.clamp((torch.abs(y_true) + torch.abs(y_point)) / 2, min=eps)
    return _format_return(_aggregate((num / denom) * 100, aggregate_method))

def smdape(y_true, y_pred, point_method="mean", aggregate_method="mean", eps=1e-8):
    """
    Symmetric Median Absolute Percentage Error (sMdAPE).
    
    .. math::
        \\text{sMdAPE} = \\text{median}\\left( 100 \\times \\frac{|y_i - \\hat{y}_i|}{\\max((|y_i| + |\\hat{y}_i|) / 2, \\epsilon)} \\right)
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_point = _get_point_forecast(y_pred, point_method)
    num = torch.abs(y_point - y_true)
    denom = torch.clamp((torch.abs(y_true) + torch.abs(y_point)) / 2, min=eps)
    error = (num / denom) * 100
    return _format_return(_aggregate(error, "median" if aggregate_method != "none" else "none"))

def mpe(y_true, y_pred, point_method="mean", aggregate_method="mean", eps=1e-8):
    """
    Mean Percentage Error (MPE).
    
    .. math::
        \\text{MPE} = \\frac{100}{N} \\sum_{i=1}^{N} \\frac{y_i - \\hat{y}_i}{\\max(y_i, \\epsilon)}
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_point = _get_point_forecast(y_pred, point_method)
    error = (y_true - y_point) / torch.clamp(y_true, min=eps)
    return _format_return(_aggregate(error * 100, aggregate_method))

def opl(y_true, y_pred, point_method="mean", aggregate_method="mean"):
    """
    Optimized Point Loss (OPL). Measures if the model correctly predicted 
    the direction of change compared to the last observation.
    
    .. math::
        \text{OPL} = \frac{1}{2N} \sum_{t=1}^{N} \left| \text{sgn}(y_{t+1} - y_t) - \text{sgn}(\hat{y}_{t+1} - y_t) \right|
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_point = _get_point_forecast(y_pred, point_method)
    
    diff_true = torch.sign(y_true[1:] - y_true[:-1])
    diff_pred = torch.sign(y_point[1:] - y_true[:-1])
    
    error = torch.abs(diff_true - diff_pred) / 2
    return _format_return(_aggregate(error, aggregate_method))

def mase(y_true, y_pred, y_train, point_method="mean", aggregate_method="mean", eps=1e-8):
    """
    Mean Absolute Scaled Error (MASE).
    
    .. math::
        \\text{MASE} = \\frac{\\frac{1}{H} \\sum_{t=1}^{H} |y_t - \\hat{y}_t|}{\\max\\left(\\frac{1}{T-1} \\sum_{t=2}^{T} |y_t - y_{t-1}|, \\epsilon\\right)}
    """
    y_true, y_pred, y_train = _sync_tensors(y_true, y_pred, y_train)
    y_point = _get_point_forecast(y_pred, point_method)
    mae_forecast = torch.abs(y_true - y_point)
    diff = torch.abs(y_train[1:] - y_train[:-1])
    scale = torch.mean(diff, dim=0) 
    scale = torch.clamp(scale, min=eps)
    mase_val = mae_forecast / scale.unsqueeze(0) 
    return _format_return(_aggregate(mase_val, aggregate_method))

def rmsse(y_true, y_pred, y_train, point_method="mean", aggregate_method="mean", eps=1e-8):
    """
    Root Mean Squared Scaled Error (RMSSE).
    
    .. math::
        \\text{RMSSE} = \\sqrt{\\frac{\\frac{1}{H} \\sum_{t=1}^{H} (y_t - \\hat{y}_t)^2}{\\max\\left(\\frac{1}{T-1} \\sum_{t=2}^{T} (y_t - y_{t-1})^2, \\epsilon\\right)}}
    """
    y_true, y_pred, y_train = _sync_tensors(y_true, y_pred, y_train)
    y_point = _get_point_forecast(y_pred, point_method)
    mse_forecast = (y_true - y_point) ** 2
    diff = (y_train[1:] - y_train[:-1]) ** 2
    scale = torch.mean(diff, dim=0)
    scale = torch.clamp(scale, min=eps)
    rmsse_val = torch.sqrt(mse_forecast / scale.unsqueeze(0))
    return _format_return(_aggregate(rmsse_val, aggregate_method))

def owa(y_true, y_pred, y_train, y_naive_pred=None, point_method="mean", aggregate_method="mean", eps=1e-8):
    """
    Overall Weighted Average (OWA).
    
    .. math::
        \\text{OWA} = \\frac{1}{2} \\left( \\frac{\\text{sMAPE}}{\\max(\\text{sMAPE}_{naive}, \\epsilon)} + \\frac{\\text{MASE}}{\\max(\\text{MASE}_{naive}, \\epsilon)} \\right)
    """
    y_true, y_pred, y_train = _sync_tensors(y_true, y_pred, y_train)
    
    if y_naive_pred is None:
        # Utilize the first point of y_pred across the time dimension (dim=0)
        # and repeat it across to form a naive prediction of identical shape
        y_naive_pred = y_pred[0:1, ...].expand_as(y_pred)
    else:
        y_naive_pred = _sync_tensors(y_naive_pred)[0]

    smape_val = smape(y_true, y_pred, point_method, "none", eps)
    smape_naive = smape(y_true, y_naive_pred, point_method, "none", eps)
    
    mase_val = mase(y_true, y_pred, y_train, point_method, "none", eps)
    mase_naive = mase(y_true, y_naive_pred, y_train, point_method, "none", eps)
    
    owa_val = 0.5 * (smape_val / torch.clamp(smape_naive, min=eps)) + 0.5 * (mase_val / torch.clamp(mase_naive, min=eps))
    return _format_return(_aggregate(owa_val, aggregate_method))


# Probabilistic metrics
def pinball_loss(y_true, y_pred, quantile, aggregate_method="mean"):
    """
    Pinball Loss (Quantile Loss).
    
    .. math::
        L_q(y, \\hat{y}) = \\max(q(y - \\hat{y}), (q - 1)(y - \\hat{y}))
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_pred_q = _get_point_forecast(y_pred, point_method=quantile)
    error = y_true - y_pred_q
    loss = torch.max(quantile * error, (quantile - 1) * error)
    return _format_return(_aggregate(loss, aggregate_method))

def crps(y_true, y_pred, aggregate_method="mean"):
    """
    Continuous Ranked Probability Score (CRPS). Empirical Approximation.
    
    .. math::
        \\text{CRPS} = \\frac{1}{S} \\sum_{s=1}^{S} |y - \\hat{y}_s| - \\frac{1}{2S^2} \\sum_{s_1=1}^{S} \\sum_{s_2=1}^{S} |\\hat{y}_{s_1} - \\hat{y}_{s_2}|
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    y_true_ext = y_true.unsqueeze(-1)
    abs_diff_true = torch.mean(torch.abs(y_pred - y_true_ext), dim=-1)
    y_pred_i = y_pred.unsqueeze(-1)
    y_pred_j = y_pred.unsqueeze(-2)
    abs_diff_samples = torch.mean(torch.abs(y_pred_i - y_pred_j), dim=(-1, -2))
    crps = abs_diff_true - 0.5 * abs_diff_samples
    return _format_return(_aggregate(crps, aggregate_method))

def empirical_coverage(y_true, y_pred, alpha=0.1, aggregate_method="mean"):
    """
    Empirical Coverage (Prediction Interval Coverage Probability - PICP).
    
    .. math::
        \\text{PICP} = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbf{1}(L_i \\leq y_i \\leq U_i)
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    lower = torch.quantile(y_pred, alpha / 2, dim=-1)
    upper = torch.quantile(y_pred, 1 - (alpha / 2), dim=-1)
    inside = ((y_true >= lower) & (y_true <= upper)).float()
    return _format_return(_aggregate(inside, aggregate_method))

def winkler_score(y_true, y_pred, alpha=0.1, aggregate_method="mean"):
    """
    Winkler Score (Mean Interval Score - MIS).
    
    .. math::
        \\text{MIS} = (U_i - L_i) + \\frac{2}{\\alpha}(L_i - y_i)\\mathbf{1}(y_i < L_i) + \\frac{2}{\\alpha}(y_i - U_i)\\mathbf{1}(y_i > U_i)
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    lower = torch.quantile(y_pred, alpha / 2, dim=-1)
    upper = torch.quantile(y_pred, 1 - (alpha / 2), dim=-1)
    widths = upper - lower
    below = (y_true < lower).float()
    above = (y_true > upper).float()
    penalty_below = (2 / alpha) * (lower - y_true) * below
    penalty_above = (2 / alpha) * (y_true - upper) * above
    scores = widths + penalty_below + penalty_above
    return _format_return(_aggregate(scores, aggregate_method))

def mpiw(y_pred, alpha=0.1, aggregate_method="mean"):
    """
    Mean Prediction Interval Width (MPIW).
    
    .. math::
        \\text{MPIW} = \\frac{1}{N} \\sum_{i=1}^{N} (U_i - L_i)
    """
    y_pred = _sync_tensors(y_pred)[0]
    lower = torch.quantile(y_pred, alpha / 2, dim=-1)
    upper = torch.quantile(y_pred, 1 - (alpha / 2), dim=-1)
    widths = upper - lower
    return _format_return(_aggregate(widths, aggregate_method))

def rho_risk(y_true, y_pred, quantiles=[0.1, 0.5, 0.9], aggregate_method="mean", eps=1e-8):
    """
    Rho-Risk (Normalized Quantile Loss).
    
    .. math::
        \\rho\\text{-risk} = \\frac{2 \\sum_i L_q(y_i, \\hat{y}_i)}{\\max(\\sum_i |y_i|, \\epsilon)}
    """
    y_true, y_pred = _sync_tensors(y_true, y_pred)
    total_true = torch.clamp(torch.sum(torch.abs(y_true)), min=eps)
    
    rho_risks = []
    for q in quantiles:
        p_loss_q = pinball_loss(y_true, y_pred, q, aggregate_method="sum")
        rho_risks.append(2 * p_loss_q / total_true)
        
    avg_rho_risk = sum(rho_risks) / len(rho_risks)
    return _format_return(_aggregate(avg_rho_risk, aggregate_method))

---CELL_BOUNDARY---

import numpy as np
import pandas as pd
import torch
import time

def get_metrics(model, test_windows, y_scaler, pred_len=24, lags=(1, 24, 168), 
                num_samples=100, seed=42, point_method="mean", 
                quantiles=[0.1, 0.5, 0.9], alpha=0.1):
    """
    Iterates over test windows, generates probabilistic forecasts, computes 
    both point and probabilistic metrics, and returns them as a pandas DataFrame.
    Measures and reports the average inference time per window.
    """
    
    # Store metrics for each rolling window
    window_metrics = []
    total_inference_time = 0.0

    for i, window_df in enumerate(test_windows):
        # 1. Data Preparation (identical to get_crps)
        full_ts = TimeSeries.from_dataframe(window_df).astype(np.float32)
        full_sc = y_scaler.transform(full_ts).astype(np.float32)
        full_pc = build_past_covs_552(full_sc, lags=lags).astype(np.float32)

        full_sc = full_sc.slice_intersect(full_pc)
        full_ts = full_ts.slice_intersect(full_sc)
        full_pc = full_pc.slice_intersect(full_sc)

        # Split into past (train) and future (ground truth)
        past_true_sc = full_sc[:-pred_len]
        past_true = full_ts[:-pred_len] # Original scale needed for MASE/RMSSE
        gt_future = full_ts[-pred_len:]

        forecast_start = gt_future.start_time()
        pc_past = ts_upto(full_pc, forecast_start)

        # 2. Forecasting with Timing and Covariates check
        start_time = time.time()
        
        predict_kwargs = {}
        if getattr(model, "supports_past_covariates", True):
            predict_kwargs["past_covariates"] = pc_past
            
        fc_sc = model.predict( 
            n=pred_len,
            series=past_true_sc,
            num_samples=num_samples,
            verbose=False,
            random_state=seed,
            **predict_kwargs
        )
        
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        # 3. Inverse transform & clip
        fc = y_scaler.inverse_transform(fc_sc)
        fc = fc.with_values(np.clip(fc.all_values(), a_min=0, a_max=None))

        assert fc.time_index.equals(gt_future.time_index), \
            f"Forecast index mismatch in window {i}"

        # 4. Extract arrays and convert to shape: (time_steps, num_nodes, num_samples)
        # Note: squeeze(-1) on target/train drops the empty sample dimension
        y_pred = fc.all_values(copy=False)                             # (T, D, S)
        y_true = gt_future.all_values(copy=False).squeeze(-1)          # (T, D)
        y_train = past_true.all_values(copy=False).squeeze(-1)         # (T_train, D)

        # 5. Compute Metrics (converting to PyTorch happens implicitly inside your metric functions)
        metrics_dict = {
            # Point Metrics
            "MAE": mae(y_true, y_pred, point_method=point_method),
            "MSE": mse(y_true, y_pred, point_method=point_method),
            "RMSE": rmse(y_true, y_pred, point_method=point_method),
            "MAPE": mape(y_true, y_pred, point_method=point_method),
            "sMAPE": smape(y_true, y_pred, point_method=point_method),
            "sMdAPE": smdape(y_true, y_pred, point_method=point_method),
            "MPE": mpe(y_true, y_pred, point_method=point_method),
            "OPL": opl(y_true, y_pred, point_method=point_method),
            
            # Scaled Point Metrics
            "MASE": mase(y_true, y_pred, y_train, point_method=point_method),
            "RMSSE": rmsse(y_true, y_pred, y_train, point_method=point_method),
            "OWA": owa(y_true, y_pred, y_train, point_method=point_method),
            
            # Probabilistic Metrics
            "CRPS": crps(y_true, y_pred),
            "PICP": empirical_coverage(y_true, y_pred, alpha=alpha),
            "MIS": winkler_score(y_true, y_pred, alpha=alpha),
            "MPIW": mpiw(y_pred, alpha=alpha),
            "Rho_Risk": rho_risk(y_true, y_pred, quantiles=quantiles),
        }
        
        window_metrics.append(metrics_dict)

    # 6. Aggregate into a DataFrame
    df_all_windows = pd.DataFrame(window_metrics)
    
    # Calculate mean across all test windows and transpose for readability
    df_summary = df_all_windows.mean().to_frame(name="Average_Score")
    
    # Report average inference time per window
    df_summary.loc["Inference Time (s)", "Average_Score"] = total_inference_time / len(test_windows)
    
    return df_summary


---CELL_BOUNDARY---

from engressionts.models.darts import (
    EnHiTSModel,
    EnTransformerModel,
    EnTCNModel,
    EnBlockRNNModel,
    EnDLinearModel,
    EnBEATSModel,
)

try:
    from engressionts.models.darts.enchronos2_model import EnChronos2Model
except ImportError as e:
    EnChronos2Model = None
    print(f"Could not import EnChronos2Model: {e}")

try:
    from engressionts.models.darts.enpatchtst_fm_model import EnPatchTSTFMModel
except ImportError as e:
    EnPatchTSTFMModel = None
    print(f"Could not import EnPatchTSTFMModel: {e}")

try:
    from engressionts.models.darts.entimesfm2p5_model import EnTimesFM2p5Model
except ImportError as e:
    EnTimesFM2p5Model = None
    print(f"Could not import EnTimesFM2p5Model: {e}")

try:
    from engressionts.models.darts.entirex_model import EnTiRexModel
except ImportError as e:
    EnTiRexModel = None
    print(f"Could not import EnTiRexModel: {e}")


---CELL_BOUNDARY---

# train_series = TimeSeries.from_dataframe(df_train)
train_pc = build_past_covs_552(train_y_sc)
train_y_sc = train_y_sc.slice_intersect(train_pc)
train_pc   = train_pc.slice_intersect(train_y_sc)

---CELL_BOUNDARY---

import logging
import lightning.pytorch as pl
import time
# Lightning 2.x
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# Older Lightning versions
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

---CELL_BOUNDARY---

train_pc = train_pc.astype(np.float32)

---CELL_BOUNDARY---

from engressionts.models.darts import (
    EnBlockRNNModel,
    # EnChronos2Model,
    EnDLinearModel,
    EnBEATSModel,
    EnHiTSModel,
    EnNLinearModel,
    EnRNNModel,
    EnTCNModel,
    EnTFTModel,
    EnTiDEModel,
    EnTSMixerModel,
    EnTransformerModel,
)

SEED = 42

deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

model = EnNLinearModel(
    input_chunk_length=24,
    output_chunk_length=PRED_LEN,

    
    num_samples=2,

    
    noise_std=1.0,

    optimizer_kwargs={"lr": 1e-3},
    random_state=SEED,
    batch_size=64,

    
    n_epochs=1,

    noise_type="gaussian",
)

start = time.time()

model.fit(
    train_y_sc,
    past_covariates=train_pc,
    verbose=True,
    dataloader_kwargs={"num_workers": 0},
)

end = time.time()

print("Training time:", end - start)

results = get_metrics(
    model,
    test_windows,
    y_scaler,
    pred_len=PRED_LEN,
    lags=(1, 24, 168),

    # Darts prediction samples
    num_samples=100,

    seed=SEED,
    point_method="median",
    quantiles=[0.1, 0.5, 0.9],
    alpha=0.05,
)

results

---CELL_BOUNDARY---

get_metrics(model, test_windows, y_scaler, pred_len=PRED_LEN, lags=(1, 24, 168), 
                num_samples=100, seed=SEED, point_method="median", 
                quantiles=[0.1, 0.5, 0.9], alpha=0.05)

---CELL_BOUNDARY---

import matplotlib.pyplot as plt
lags = LAGS
pred_len=24
history_hours = 24
full_ts = TimeSeries.from_dataframe(test_windows[4]).astype(np.float32)
full_sc = y_scaler.transform(full_ts).astype(np.float32)
full_pc = build_past_covs_552(full_sc, lags=lags).astype(np.float32)

full_sc = full_sc.slice_intersect(full_pc)
full_ts = full_ts.slice_intersect(full_sc)
full_pc = full_pc.slice_intersect(full_sc)

past_true_sc = full_sc[-2*pred_len:-pred_len]
gt_future    = full_ts[-pred_len:]

forecast_start = gt_future.start_time()          
pc_past = ts_upto(full_pc, forecast_start)      

#print(f"Past Values are {past_true_sc}")
#model.model.encoder[0].reset_std(2)
# EnHiTS has no encoder/reset_seed/reset_std
fc_sc = model.predict(
    pred_len,
    series=past_true_sc,
    past_covariates=pc_past,
    num_samples=100,
    verbose=False,
)
print(fc_sc.shape)
fc = y_scaler.inverse_transform(fc_sc)
fc = fc.with_values(np.clip(fc.all_values(), a_min=0, a_max=None))

print(fc.shape)

forecast_unscaled = fc

# Ensure training data is also unscaled for plotting
# 'train' should be your original unscaled training TimeSeries
train_unscaled = full_ts[-2*pred_len:-pred_len]
actual_unscaled = gt_future

first_10_names = list(forecast_unscaled.components[:16])

# Subsetting
forecast_sub = forecast_unscaled[first_10_names]
actual_sub = actual_unscaled[first_10_names]
train_sub = train_unscaled[first_10_names]

import matplotlib.pyplot as plt

# 1. Enforce default matplotlib style
plt.style.use('default')

# Create figure and axes explicitly to manage the layout better
fig, axes = plt.subplots(4, 4, figsize=(18, 25))
axes = axes.flatten()

# Add a main title to the overall figure
fig.suptitle("Solar Dataset, First 16 Nodes: Test Window 5", fontsize=18, y=0.96)

for i in range(16):
    ax = axes[i]
    plt.sca(ax)  # Set current axis for Darts
    
    # Plot Last 24 points of Training Data
    train_sub.univariate_component(i).plot(
        label="Historical Data (Last 24)", 
        color="black", 
        linestyle="--", 
        linewidth=1.5
    )
    
    # Plot Ground Truth (Actuals)
    actual_sub.univariate_component(i).plot(
        label="Ground Truth", 
        color="black", 
        linewidth=2,
        zorder=10 
    )
    
    # Forecast Layer: 95% CI + Median
    forecast_sub.univariate_component(i).plot(
        central_quantile=0.5,    
        low_quantile=0.025,      # 2.5th percentile
        high_quantile=0.975,     # 97.5th percentile
        label="Median Forecast", # Updated label
        color="blue",
        alpha=0.25                # Single interval with 0.3 opacity
    )
    
    ax.set_title(f"Node: {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    
    # Darts sometimes automatically adds a legend; we remove it here to keep subplots clean
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# 2. Extract handles and labels for the unified legend
handles, labels = ax.get_legend_handles_labels()

# Deduplicate in case multiple calls generated duplicate labels
by_label = dict(zip(labels, handles))

# 3. Create a single figure-level legend at the bottom center
fig.legend(
    by_label.values(), 
    by_label.keys(), 
    loc='lower center', 
    ncol=3,                     # Reduced to 3 columns to match the 3 legend items
    fontsize='large', 
    bbox_to_anchor=(0.5, 0.035) 
)

# 4. Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 0.96]) 
plt.savefig("Solar_Forecasts.pdf", format='pdf')
plt.show()

---CELL_BOUNDARY---

SEED = 42
deterministic.init_all(SEED)
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

---CELL_BOUNDARY---

import time
import logging
import os
import gc
import torch
import pandas as pd
from engressionts.models.darts import (
    EnBlockRNNModel, EnDLinearModel, EnBEATSModel, 
    EnHiTSModel, EnNLinearModel, EnRNNModel, EnTCNModel, EnTFTModel, 
    EnTiDEModel, EnTSMixerModel, EnTransformerModel
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# List of all models to run (Display Name, Class, Extra Init Parameters)
MODELS_TO_RUN_RAW = [
    ("EnBlockRNNModel", EnBlockRNNModel, {}),
    ("EnBlockLSTM", EnBlockRNNModel, {"model": "LSTM"}),
    ("EnBlockGRU", EnBlockRNNModel, {"model": "GRU"}),
    ("EnDLinearModel", EnDLinearModel, {}),
    ("EnBEATSModel", EnBEATSModel, {}),
    ("EnHiTSModel", EnHiTSModel, {}),
    ("EnNLinearModel", EnNLinearModel, {}),
    ("EnRNNModel", EnRNNModel, {}),
    ("EnLSTM", EnRNNModel, {"model": "LSTM"}),
    ("EnGRU", EnRNNModel, {"model": "GRU"}),
    ("EnTCNModel", EnTCNModel, {}),
    ("EnTFTModel", EnTFTModel, {}),
    ("EnTiDEModel", EnTiDEModel, {}),
    ("EnTSMixerModel", EnTSMixerModel, {}),
    ("EnTransformerModel", EnTransformerModel, {}),
    # Darts Foundation Models (Tested Zero-Shot using n_epochs=0 to avoid OOM)
    ("EnChronos2Model", EnChronos2Model, {"n_epochs": 0}),
    ("EnPatchTSTFMModel", EnPatchTSTFMModel, {"n_epochs": 0}),
    ("EnTimesFM2p5Model", EnTimesFM2p5Model, {"n_epochs": 0}),
    ("EnTiRexModel", EnTiRexModel, {"n_epochs": 0, "accept_license": True}),
]

# Filter out models that were not imported (are None)
MODELS_TO_RUN = [
    (name, cls, kwargs) for name, cls, kwargs in MODELS_TO_RUN_RAW
    if cls is not None
]

BASE_KWARGS = {
    "output_chunk_length": PRED_LEN, 
    "num_samples_train": 2,
    "noise_std": 1,
    "optimizer_kwargs": {"lr": 1e-3},
    "random_state": SEED,
    "batch_size": 64,
    "n_epochs": 30,
    "noise_type": "gaussian"
}

CSV_FILENAME = "Solar_EngressionTS_uniform_metrics.csv"

# Load existing results to support resume
if os.path.exists(CSV_FILENAME):
    logger.info(f"Found existing results file '{CSV_FILENAME}'. Loading to resume progress...")
    results_df = pd.read_csv(CSV_FILENAME)
    # Load completed model names
    completed_models = set(results_df["Model"].tolist()) if "Model" in results_df.columns else set()
    all_results = results_df.to_dict(orient="records")
else:
    logger.info("No existing results file found. Starting fresh...")
    completed_models = set()
    all_results = []

total_models = len(MODELS_TO_RUN)

for idx, (model_name, model_class, extra_kwargs) in enumerate(MODELS_TO_RUN, 1):
    
    if model_name in completed_models:
        logger.info(f"[{idx}/{total_models}] Skipping {model_name} - already completed.")
        continue
        
    logger.info(f"[{idx}/{total_models}] Starting {model_name}...")
    
    # 1. Clean up GPU memory before starting a new model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Dynamic Model Configurations
    model_kwargs = BASE_KWARGS.copy()
    model_kwargs.update(extra_kwargs) # Apply model-specific overrides (like n_epochs=0)
    
    # Dynamically map parameters based on constructor signature to prevent ValueError
    import inspect
    sig = inspect.signature(model_class.__init__)
    if "num_samples_train" not in sig.parameters and "num_samples" in sig.parameters:
        if "num_samples_train" in model_kwargs:
            model_kwargs["num_samples"] = model_kwargs.pop("num_samples_train")
            
    if "num_samples_train" in model_kwargs and "num_samples_train" not in sig.parameters:
        del model_kwargs["num_samples_train"]
        
    model_kwargs["input_chunk_length"] = 30 if model_name == "EnTCNModel" else 24
    
    if model_name == "EnTFTModel":
        model_kwargs["add_relative_index"] = True
        
    if model_name == "EnChronos2Model":
        model_kwargs["batch_size"] = 2 
        
    # 3. Dynamic Fit Configurations
    fit_kwargs = {
        "verbose": True,
        "dataloader_kwargs": {"num_workers": 0},
    }
    
    # Omit past covariates for models that do not support them
    if model_name in ["EnRNNModel", "EnLSTM", "EnGRU", "EnPatchTSTFMModel", "EnTimesFM2p5Model", "EnTiRexModel"]:
        pass
    else:
        fit_kwargs["past_covariates"] = train_pc

    try:
        model = model_class(**model_kwargs)
        
        logger.info(f"[{idx}/{total_models}] Training {model_name}...")
        start_time = time.time()
        
        model.fit(
            train_y_sc,
            **fit_kwargs
        )
        
        training_time = time.time() - start_time
        logger.info(f"[{idx}/{total_models}] {model_name} trained/initialized in {training_time:.2f} seconds.")
        
        logger.info(f"[{idx}/{total_models}] Evaluating {model_name}...")
        metrics_df = get_metrics(
            model, test_windows, y_scaler, pred_len=PRED_LEN, lags=(1, 24, 168), 
            num_samples=100, seed=SEED, point_method="median", 
            quantiles=[0.1, 0.5, 0.9], alpha=0.05
        )
        
        # 4. Restructure results aggregation in tabular format (row-per-model)
        row = {
            "Model": model_name,
            "Batch Size": model_kwargs.get("batch_size", 64),
            "Max Steps/Epochs": model_kwargs.get("n_epochs", 30),
            "Training Samples": model_kwargs.get("num_samples_train", model_kwargs.get("num_samples", 2)),
            "Prediction Samples": 100,
            "Noise Std": model_kwargs.get("noise_std", 1.0),
            "Noise Type": model_kwargs.get("noise_type", "gaussian"),
            "Input Length": model_kwargs.get("input_chunk_length", 24),
            "Forecast Horizon": PRED_LEN,
            "Training Time (s)": training_time,
            "Inference Time (s)": metrics_df.loc["Inference Time (s)", "Average_Score"] if "Inference Time (s)" in metrics_df.index else 0.0,
        }
        
        # Add other metric scores dynamically
        for metric_name in metrics_df.index:
            if metric_name not in ["Inference Time (s)", "Inference Time"]:
                row[metric_name] = metrics_df.loc[metric_name, "Average_Score"]
                
        all_results.append(row)
        
        # Save after every model in tabular format
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(CSV_FILENAME, index=False)
        logger.info(f"[{idx}/{total_models}] Saved results for {model_name} in tabular format to {CSV_FILENAME}.\n")
        
    except Exception as e:
        logger.error(f"[{idx}/{total_models}] FAILED on {model_name}. Error: {str(e)}\n")
        continue

logger.info(f"Job finished. Final consolidated metrics in tabular format saved to {CSV_FILENAME}.")

---CELL_BOUNDARY---

vals = fc.all_values()

print("fc.shape:", fc.shape)
print("all_values shape:", vals.shape)

print("timesteps =", vals.shape[0])
print("nodes      =", vals.shape[1])

if vals.ndim == 3:
    print("samples    =", vals.shape[2])
elif vals.ndim == 2:
    print("No sample dimension present (deterministic output)")
else:
    print("Unexpected shape!")

---CELL_BOUNDARY---

lst = []

start = time.time()
for i in range(10):
    lst.append(
        get_crps(
            model,
            test_windows,
            y_scaler,
            seed=SEED + i,
        )
    )
end = time.time()

print("Inference time:", (end - start))
print(np.mean(lst))

---CELL_BOUNDARY---

get_metrics(model, test_windows, y_scaler, pred_len=24, lags=(1, 24, 168), 
                num_samples=100, seed=SEED, point_method="median", 
                quantiles=[0.1, 0.5, 0.9], alpha=0.05)

---CELL_BOUNDARY---

lst = []
seeds = [60, 167, 221, 432, 264, 253, 266, 496, 151, 99]

from tqdm import tqdm

for i in tqdm(seeds):
    lst.append(
        get_crps(
            model,
            test_windows,
            y_scaler,
            seed=i,
        )
    )

print(np.mean(lst), np.std(lst))

---CELL_BOUNDARY---

import pickle
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np

# --- 1. Search Space ---
search_space = {
    "std": hp.uniform("std", 0.5, 3.0),
    "engression_m": hp.quniform("engression_m", 2, 8, 2),
    "learning_rate": hp.qloguniform(
        "learning_rate", np.log(1e-5), np.log(1e-2), 1e-5
    ),
    "batch_size": hp.quniform("batch_size", 32, 128, 32),
}

# --- 2. Objective Function ---
def objective(params):

    SEED = 42
    deterministic.init_all(SEED)
    pl.seed_everything(SEED, workers=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    try:
        model = EnHiTSModel(
            input_chunk_length=24,
            output_chunk_length=24,
            noise_std=params["std"],
            num_samples_engression=int(params["engression_m"]),
            optimizer_kwargs={"lr": params["learning_rate"]},
            batch_size=int(params["batch_size"]),
            random_state=SEED,
            n_epochs=30,
            noise_dist="uniform",
        )

        model.fit(
            train_y_sc,
            past_covariates=train_pc,
            verbose=True,
            dataloader_kwargs={"num_workers": 0},
        )

        crps_list = []
        for i in range(10):
            crps_list.append(
                get_crps(
                    model,
                    test_windows,
                    y_scaler,
                    seed=SEED + i,
                )
            )

        mean_crps = np.mean(crps_list)
        std_crps = np.std(crps_list)

        print(f"Got Result {mean_crps} +/- {std_crps} for {params}")

        return {
            "loss": mean_crps,
            "status": STATUS_OK,
            "crps_list": crps_list,
        }

    except Exception as e:
        print(f"Trial failed: {e}")
        return {"loss": 1e6, "status": STATUS_OK}


# --- 3. Hyperopt Loop ---
trials_step = 5
max_trials = 50
trials_file = "hyperopt_trials_enhits.pkl"

try:
    with open(trials_file, "rb") as f:
        trials = pickle.load(f)
    print(f"Found existing trials. Resuming from {len(trials.trials)} runs.")
except:
    trials = Trials()

for i in range(len(trials.trials) + trials_step, max_trials + trials_step, trials_step):
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=i,
        trials=trials,
        rstate=np.random.default_rng(SEED),
    )

    with open(trials_file, "wb") as f:
        pickle.dump(trials, f)

    print(f"Checkpoint saved at {i} trials. Best so far: {best}")

---CELL_BOUNDARY---

