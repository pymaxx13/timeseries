from typing import Optional, Union, List
import torch
import torch.nn as nn
from neuralforecast.losses.pytorch import BasePointLoss
from neuralforecast.models.patchtst import PatchTST_backbone

from engressionts.base.base_engression import NFEngressionBaseModel
from engressionts.losses.energy_score import EnergyScoreLoss


class EnPatchTST(NFEngressionBaseModel):
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    EXOGENOUS_CAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        encoder_layers: int = 3,
        n_heads: int = 16,
        hidden_size: int = 128,
        linear_hidden_size: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        revin: bool = True,
        revin_affine: bool = False,
        revin_subtract_last: bool = True,
        activation: str = "gelu",
        res_attention: bool = True,
        batch_normalization: bool = False,
        learn_pos_embed: bool = True,
        loss: Optional[Union[BasePointLoss, nn.Module]] = None,
        valid_loss: Optional[Union[BasePointLoss, nn.Module]] = None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_monitor: str = "ptl/val_loss",
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: Optional[int] = 1024,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: float = 0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        noise_std: float = 1.0,
        noise_type: str = "gaussian",
        num_samples_train: int = 20, num_samples=None,
        **trainer_kwargs,
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
            noise_std=noise_std,
            noise_type=noise_type,
            num_samples_train=num_samples_train, num_samples=num_samples,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_monitor=val_monitor,
            **trainer_kwargs,
        )

        patch_len = min(input_size + stride, patch_len)
        c_out = self.loss.outputsize_multiplier

        c_in = 1
        padding_patch = "end"
        pretrain_head = False
        norm = "BatchNorm"
        pe = "zeros"
        d_k = None
        d_v = None
        store_attn = False
        head_type = "flatten"
        individual = False
        max_seq_len = 1024
        key_padding_mask = "auto"
        padding_var = None
        attn_mask = None

        self.model = PatchTST_backbone(
            c_in=c_in,
            c_out=c_out,
            input_size=input_size,
            h=h,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=activation,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=batch_normalization,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pos_embed,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=revin_affine,
            subtract_last=revin_subtract_last,
        )

    def forward(self, windows_batch):
        x = windows_batch["insample_y"]  # [B * M, L, 1]
        x = self.noise_layer(x)  # Noise injection point
        x = x.permute(0, 2, 1)  # [B * M, 1, L]
        x = self.model(x)  # [B * M, 1, H * c_out]
        forecast = x.reshape(x.shape[0], self.h, -1)  # [B * M, H, 1]
        return forecast
