#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:20:58 2023

@author: louis
"""

import torch
from torch import nn
from model.utils.abs_models import AbsSpeechModel, SignalDomain

from espnet2.enh.separator.tfgridnetv3_separator import TFGridNetV3


class TFGridNet(AbsSpeechModel):
    def __init__(
        self,
        metrics: list[nn.Module] = [],
        n_layers: int = 4,  # B for USDnet
        lstm_hidden_units: int = 192,  # H for USDnet
        attn_n_head: int = 4,  # L for USDnet
        attn_qk_output_channel: int = 2,  # E for USDnet
        emb_dim: int = 48,  # D for USDnet
        emb_ks: int = 4,  # I
        emb_hs: int = 4,  # J
    ):
        super().__init__(metrics=metrics, crop_input_to_target=True)
        self.original_tfgridnet = TFGridNetV3(
            input_dim=None,
            n_srcs=1,
            n_imics=1,
            n_layers=n_layers,  # B for USDnet
            lstm_hidden_units=lstm_hidden_units,  # H for USDnet
            attn_n_head=attn_n_head,  # L for USDnet
            attn_qk_output_channel=attn_qk_output_channel,  # E for USDnet
            emb_dim=emb_dim,  # D for USDnet
            emb_ks=emb_ks,  # I
            emb_hs=emb_hs,  # J
            activation="prelu",
            eps=1.0e-5,
        )
        self.l1_loss = nn.L1Loss(reduction="mean")

    def forward(self, y):
        normalization_factor = torch.std(y, (1, 2), keepdims=True)
        normalized_y = y / normalization_factor
        Y = self.stft_module(normalized_y)
        # B C F T -> B T F
        Y = Y[..., 0, :, :].transpose(-1, -2)
        ilens = None
        [S_hat], _, _ = self.original_tfgridnet(Y, ilens=None)
        # we need to pass along the normalization factor for the model to work
        return S_hat[:, None, :, :].transpose(-2, -1), normalization_factor

    def internal_loss(self, pred, s):
        # corresponds to equation 11 of 10.1109/TASLP.2023.3304482 (used for WSJ0CAM-DEREVERB)
        # We normalize the sample variance of the time-domain mixture to one
        # and scale each clean source using the same scaling factor during training.
        S_hat, normalization_factor = pred
        s_scaled = s / normalization_factor
        S = self.stft_module(s_scaled)
        s_hat = self.get_time(pred, length=s.size(-1))
        loss_magnitude = self.l1_loss(s_hat, s)
        loss_stft = self.l1_loss(S_hat.abs(), S.abs())
        return loss_magnitude + loss_stft

    def get_stft(self, pred, **kwargs):
        pred_stft, normalization_factor = pred
        return pred_stft

    def get_time(self, pred, length=None):
        pred_stft, normalization_factor = pred
        return self.istft_module(pred_stft, length=length)


if __name__ == "__main__":
    batch = torch.randn(2, 1, 48000, device="cuda"), torch.randn(2, 1, 48000, device="cuda")
    model = TFGridNet().to(device="cuda")
    s_hat, normalization = model.forward(batch[0])
    # res=model.training_step(batch, batch_idx=0)
