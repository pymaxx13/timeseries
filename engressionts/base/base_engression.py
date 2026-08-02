import math
from typing import Optional, List, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import neuralforecast
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import BasePointLoss

from engressionts.losses.energy_score import energy_score_loss, EnergyScoreLoss
from engressionts.noise import NOISE_REGISTRY


class EngressionPLModule(PLForecastingModule):
    def __init__(
        self,
        noise_std: float = 1.0,
        noise_type: str = "gaussian",
        num_samples: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.noise_std = noise_std
        self.noise_type = noise_type
        self.num_samples = num_samples

        self.noise_layer = self._build_noise_layer()

    def _build_noise_layer(self):
        try:
            noise_cls = NOISE_REGISTRY[self.noise_type]
        except KeyError as exc:
            raise ValueError(f"Unknown Engression noise type: {self.noise_type}") from exc

        return noise_cls(std=self.noise_std)

    def _repeat_tensor(self, tensor):
        return (
            tensor.repeat_interleave(self.num_samples, dim=0)
            if tensor is not None
            else None
        )

    def training_step(self, batch, batch_idx):
        """Train on ``M`` noise-perturbed forecasts using the Energy Score."""
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            _,
            future_target,
        ) = batch

        batch_size = future_target.shape[0]

        y_hat = self._produce_train_output(
            (
                self._repeat_tensor(past_target),
                self._repeat_tensor(past_covariates),
                self._repeat_tensor(historic_future_covariates),
                self._repeat_tensor(future_covariates),
                self._repeat_tensor(static_covariates),
            )
        )

        if y_hat.shape[-1] != 1:
            raise ValueError(
                "Engression models currently require `likelihood=None`, because the "
                "Energy Score is computed from forecast samples."
            )

        y_hat = y_hat.squeeze(-1)
        samples = y_hat.view(
            batch_size,
            self.num_samples,
            y_hat.shape[1],
            y_hat.shape[2],
        ).permute(1, 0, 2, 3)

        loss = energy_score_loss(samples, future_target)
        self.log(
            "energy_score_train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Enable input noise while Darts generates prediction samples."""
        noise_layer_was_training = self.noise_layer.training
        self.noise_layer.train()
        try:
            return super().predict_step(batch, batch_idx, dataloader_idx)
        finally:
            self.noise_layer.train(noise_layer_was_training)


class NFEngressionBaseModel(BaseModel):
    def __init__(
        self,
        h: int,
        input_size: int,
        loss: Optional[Union[BasePointLoss, nn.Module]] = None,
        valid_loss: Optional[Union[BasePointLoss, nn.Module]] = None,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: Optional[int] = None,
        start_padding_enabled: bool = False,
        noise_std: float = 1.0,
        noise_type: str = "gaussian",
        num_samples: int = 20,
        **kwargs,
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
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            **kwargs,
        )
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.num_samples = num_samples

        self.noise_layer = self._build_noise_layer()

    def _build_noise_layer(self):
        try:
            noise_cls = NOISE_REGISTRY[self.noise_type]
        except KeyError as exc:
            raise ValueError(f"Unknown Engression noise type: {self.noise_type}") from exc

        return noise_cls(std=self.noise_std)

    def _repeat_tensor(self, tensor, dim=0):
        if tensor is None:
            return None
        return tensor.repeat_interleave(self.num_samples, dim=dim)

    def training_step(self, batch, batch_idx):
        if self.RECURRENT:
            self.h = self.h_train

        y_idx = batch["y_idx"]

        # Create windows
        windows_temporal, static, static_cols, final_condition, sample_weight_windows, temporal_cols = (
            self._create_windows(batch, step="train")
        )
        final_condition = self._shard_multivariate_windows(final_condition)
        n_windows = len(final_condition)

        # Slice batch windows
        if self.windows_batch_size is not None:
            if n_windows < self.windows_batch_size:
                w_idxs = torch.randint(
                    0,
                    n_windows,
                    size=(self.windows_batch_size,),
                    device=windows_temporal.device,
                )
            else:
                w_idxs = torch.randperm(n_windows, device=windows_temporal.device)[
                    : self.windows_batch_size
                ]
        else:
            w_idxs = torch.arange(n_windows, device=windows_temporal.device)

        windows = self._sample_windows(
            windows_temporal=windows_temporal,
            static=static,
            static_cols=static_cols,
            temporal_cols=temporal_cols,
            w_idxs=w_idxs,
            final_condition=final_condition,
            sample_weight=sample_weight_windows,
        )

        original_outsample_y = torch.clone(
            windows["temporal"][:, self.input_size :, y_idx]
        )
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        sample_weight = windows.get("sample_weight", None)
        if sample_weight is not None:
            outsample_mask = outsample_mask * sample_weight

        # Repeat parsed window batches M times along batch dim (dim=0)
        repeated_insample_y = self._repeat_tensor(insample_y, dim=0)
        repeated_insample_mask = self._repeat_tensor(insample_mask, dim=0)
        repeated_hist_exog = self._repeat_tensor(hist_exog, dim=0)
        repeated_futr_exog = self._repeat_tensor(futr_exog, dim=0)
        if stat_exog is not None and not self.MULTIVARIATE:
            repeated_stat_exog = self._repeat_tensor(stat_exog, dim=0)
        else:
            repeated_stat_exog = stat_exog

        windows_batch = dict(
            insample_y=repeated_insample_y,
            insample_mask=repeated_insample_mask,
            futr_exog=repeated_futr_exog,
            hist_exog=repeated_hist_exog,
            stat_exog=repeated_stat_exog,
        )

        # Model predictions
        output = self(windows_batch)
        output = self.loss.domain_map(output)

        # Unnormalize to raw scale
        y_loc, y_scale = self._get_loc_scale(y_idx)
        repeated_y_loc = self._repeat_tensor(y_loc, dim=0)
        repeated_y_scale = self._repeat_tensor(y_scale, dim=0)
        unnormalized_output = self.scaler.inverse_transform(z=output, x_scale=repeated_y_scale, x_shift=repeated_y_loc)
        unnormalized_target = original_outsample_y

        B = original_outsample_y.shape[0]
        samples = unnormalized_output.view(B, self.num_samples, self.h, -1).permute(1, 0, 2, 3)

        # Compute Energy Score loss
        loss = self.loss(y=unnormalized_target, y_hat=samples, mask=outsample_mask)

        if torch.isnan(loss):
            raise Exception("Loss is NaN, training stopped.")

        train_loss_log = loss.detach().item()
        self.log(
            "train_loss",
            train_loss_log,
            batch_size=B,
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, train_loss_log))

        self.h = self.horizon_backup

        return loss

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        windows_temporal, static, static_cols, final_condition, sample_weight_windows, temporal_cols = (
            self._create_windows(batch, step="val")
        )
        n_windows = len(final_condition)
        y_idx = batch["y_idx"]

        # Number of windows in batch
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        valid_losses = []
        batch_sizes = []
        for i in range(n_batches):
            w_idxs = torch.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows), device=windows_temporal.device
            )
            windows = self._sample_windows(
                windows_temporal,
                static,
                static_cols,
                temporal_cols,
                w_idxs=w_idxs,
                final_condition=final_condition,
                sample_weight=sample_weight_windows,
            )
            original_outsample_y = torch.clone(
                windows["temporal"][:, self.input_size :, y_idx]
            )

            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            (
                insample_y,
                insample_mask,
                _,
                outsample_mask,
                hist_exog,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)

            sample_weight = windows.get("sample_weight", None)
            if sample_weight is not None:
                outsample_mask = outsample_mask * sample_weight

            # Repeat variables along batch dimension
            repeated_insample_y = self._repeat_tensor(insample_y, dim=0)
            repeated_insample_mask = self._repeat_tensor(insample_mask, dim=0)
            repeated_hist_exog = self._repeat_tensor(hist_exog, dim=0)
            repeated_futr_exog = self._repeat_tensor(futr_exog, dim=0)
            if stat_exog is not None and not self.MULTIVARIATE:
                repeated_stat_exog = self._repeat_tensor(stat_exog, dim=0)
            else:
                repeated_stat_exog = stat_exog

            # We need to predict recursively if RECURRENT
            if self.RECURRENT:
                output_batch = self._validate_step_recurrent_batch(
                    insample_y=repeated_insample_y,
                    insample_mask=repeated_insample_mask,
                    futr_exog=repeated_futr_exog,
                    hist_exog=repeated_hist_exog,
                    stat_exog=repeated_stat_exog,
                    y_idx=y_idx,
                )
            else:
                windows_batch = dict(
                    insample_y=repeated_insample_y,
                    insample_mask=repeated_insample_mask,
                    futr_exog=repeated_futr_exog,
                    hist_exog=repeated_hist_exog,
                    stat_exog=repeated_stat_exog,
                )
                output_batch = self(windows_batch)

            output_batch = self.loss.domain_map(output_batch)

            y_loc, y_scale = self._get_loc_scale(y_idx)
            repeated_y_loc = self._repeat_tensor(y_loc, dim=0)
            repeated_y_scale = self._repeat_tensor(y_scale, dim=0)
            unnormalized_output = self.scaler.inverse_transform(z=output_batch, x_scale=repeated_y_scale, x_shift=repeated_y_loc)

            B = original_outsample_y.shape[0]
            samples = unnormalized_output.view(B, self.num_samples, self.h, -1).permute(1, 0, 2, 3)

            # Compute Energy Score loss
            loss_batch = self.valid_loss(y=original_outsample_y, y_hat=samples, mask=outsample_mask)
            valid_losses.append(loss_batch)
            batch_sizes.append(B)

        valid_loss = torch.stack(valid_losses)
        batch_sizes = torch.tensor(batch_sizes, device=valid_loss.device)
        batch_size = torch.sum(batch_sizes)
        valid_loss = torch.sum(valid_loss * batch_sizes) / batch_size

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        valid_loss_log = valid_loss.detach()
        self.log(
            "valid_loss",
            valid_loss_log.item(),
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
        )
        self.validation_step_outputs.append(valid_loss_log)
        return valid_loss

    def _predict_step_direct_batch(
        self, insample_y, insample_mask, hist_exog, futr_exog, stat_exog, y_idx
    ):
        # Repeat parsed window batches M times along batch dim (dim=0)
        repeated_insample_y = self._repeat_tensor(insample_y, dim=0)
        repeated_insample_mask = self._repeat_tensor(insample_mask, dim=0)
        repeated_hist_exog = self._repeat_tensor(hist_exog, dim=0)
        repeated_futr_exog = self._repeat_tensor(futr_exog, dim=0)
        if stat_exog is not None and not self.MULTIVARIATE:
            repeated_stat_exog = self._repeat_tensor(stat_exog, dim=0)
        else:
            repeated_stat_exog = stat_exog

        windows_batch = dict(
            insample_y=repeated_insample_y,
            insample_mask=repeated_insample_mask,
            futr_exog=repeated_futr_exog,
            hist_exog=repeated_hist_exog,
            stat_exog=repeated_stat_exog,
        )

        # Model Predictions
        output_batch = self(windows_batch)
        output_batch = self.loss.domain_map(output_batch)

        # Inverse normalization using y_loc and y_scale context stats
        y_loc, y_scale = self._get_loc_scale(y_idx)
        repeated_y_loc = self._repeat_tensor(y_loc, dim=0)
        repeated_y_scale = self._repeat_tensor(y_scale, dim=0)
        unnormalized_output = self.scaler.inverse_transform(z=output_batch, x_scale=repeated_y_scale, x_shift=repeated_y_loc)

        B = insample_y.shape[0]
        # Reshape to separate batch and sample dimensions
        samples = unnormalized_output.view(B, self.num_samples, self.h, -1)  # [B, M, H, D]

        # Aggregate prediction samples to match NeuralForecast's expected output schema
        if hasattr(self.loss, "quantiles") and self.loss.quantiles is not None and len(self.loss.quantiles) > 0:
            qs = self.loss.quantiles.to(samples.device)
            quantile_forecasts = torch.quantile(samples, q=qs, dim=1)  # [len(qs), B, H, D]
            y_hat = quantile_forecasts.permute(1, 2, 3, 0)  # [B, H, D, len(qs)]
        else:
            y_hat = samples.mean(dim=1).unsqueeze(-1)  # [B, H, D, 1]

        n_outputs = len(self.loss.output_names)

        if not self.MULTIVARIATE:
            y_hat = y_hat.squeeze(2)  # [B, H, n_outputs]
            if n_outputs == 1:
                y_hat = y_hat.squeeze(-1)  # [B, H]
        else:
            if n_outputs == 1:
                y_hat = y_hat.squeeze(-1)  # [B, H, D]

        return y_hat

    def _predict_step_recurrent_batch(
        self, insample_y, insample_mask, futr_exog, hist_exog, stat_exog, y_idx
    ):
        # Remember state in network and set horizon to 1
        self.rnn_state = None
        self.maintain_state = True
        self.h = 1

        # Repeat parsed window batches M times along batch dim (dim=0)
        repeated_insample_y = self._repeat_tensor(insample_y, dim=0)
        repeated_insample_mask = self._repeat_tensor(insample_mask, dim=0)
        repeated_hist_exog = self._repeat_tensor(hist_exog, dim=0)
        repeated_futr_exog = self._repeat_tensor(futr_exog, dim=0)
        if stat_exog is not None and not self.MULTIVARIATE:
            repeated_stat_exog = self._repeat_tensor(stat_exog, dim=0)
        else:
            repeated_stat_exog = stat_exog

        # Temporarily repeat scaler stats to batch size B * M
        original_x_scale = None
        original_x_shift = None
        if hasattr(self.scaler, "x_scale") and self.scaler.x_scale is not None:
            original_x_scale = self.scaler.x_scale
            original_x_shift = self.scaler.x_shift
            self.scaler.x_scale = self._repeat_tensor(original_x_scale, dim=0)
            self.scaler.x_shift = self._repeat_tensor(original_x_shift, dim=0)

        y_hat_temp = torch.zeros(
            (repeated_insample_y.shape[0], self.predict_horizon, repeated_insample_y.shape[2], 1),
            device=insample_y.device,
            dtype=insample_y.dtype,
        )

        curr_insample_y = repeated_insample_y
        curr_insample_mask = repeated_insample_mask

        for tau in range(self.predict_horizon):
            if tau == 0:
                hist_exog_current = hist_exog[:, : self.input_size] if self.hist_exog_size > 0 else None
                futr_exog_current = futr_exog[:, : self.input_size] if self.futr_exog_size > 0 else None
                hist_exog_current = self._repeat_tensor(hist_exog_current, dim=0)
                futr_exog_current = self._repeat_tensor(futr_exog_current, dim=0)
            else:
                hist_exog_current = hist_exog[:, self.input_size + tau - 1].unsqueeze(1) if self.hist_exog_size > 0 else None
                futr_exog_current = futr_exog[:, self.input_size + tau - 1].unsqueeze(1) if self.futr_exog_size > 0 else None
                hist_exog_current = self._repeat_tensor(hist_exog_current, dim=0)
                futr_exog_current = self._repeat_tensor(futr_exog_current, dim=0)

            # Reuses standard recursive step calculations (_predict_step_recurrent_single)
            y_hat_step, curr_insample_y = self._predict_step_recurrent_single(
                insample_y=curr_insample_y,
                insample_mask=curr_insample_mask if tau == 0 else None,
                hist_exog=hist_exog_current,
                futr_exog=futr_exog_current,
                stat_exog=repeated_stat_exog,
                y_idx=y_idx,
            )

            y_hat_temp[:, tau] = y_hat_step if y_hat_step.ndim == 3 else y_hat_step.unsqueeze(-1)

        # Restore original scaler stats
        if original_x_scale is not None:
            self.scaler.x_scale = original_x_scale
            self.scaler.x_shift = original_x_shift

        # Reset state and horizon
        self.maintain_state = False
        self.rnn_state = None
        self.h = self.horizon_backup

        # Reshape to [B, M, H, D]
        samples = y_hat_temp.squeeze(-1).view(insample_y.shape[0], self.num_samples, self.predict_horizon, insample_y.shape[2])

        # Aggregate prediction samples to match NeuralForecast's expected output schema
        if hasattr(self.loss, "quantiles") and self.loss.quantiles is not None and len(self.loss.quantiles) > 0:
            qs = self.loss.quantiles.to(samples.device)
            quantile_forecasts = torch.quantile(samples, q=qs, dim=1)  # [len(qs), B, H, D]
            y_hat = quantile_forecasts.permute(1, 2, 3, 0)  # [B, H, D, len(qs)]
        else:
            y_hat = samples.mean(dim=1).unsqueeze(-1)  # [B, H, D, 1]

        n_outputs = len(self.loss.output_names)

        if not self.MULTIVARIATE:
            y_hat = y_hat.squeeze(2)  # [B, H, n_outputs]
            if n_outputs == 1:
                y_hat = y_hat.squeeze(-1)  # [B, H]
        else:
            if n_outputs == 1:
                y_hat = y_hat.squeeze(-1)  # [B, H, D]

        return y_hat
