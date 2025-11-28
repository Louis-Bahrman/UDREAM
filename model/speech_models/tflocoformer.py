#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:20:58 2023

@author: louis
"""

import torch
from torch import nn
from model.utils.abs_models import AbsSpeechModel, SignalDomain

from model.speech_models.tflocoformer_separator import TFLocoformerSeparator
from auraloss.freq import MultiResolutionSTFTLoss


class TFLocoformer(AbsSpeechModel):
    def __init__(
        self,
        metrics: list[nn.Module] = [],
        n_layers: int = 6,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        # ffn related
        ffn_type: str | list = ["swiglu_conv1d"],
        ffn_hidden_dim: int | list = [384],
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # others
        eps: float = 1.0e-5,
    ):
        super().__init__(metrics=metrics, crop_input_to_target=True)
        self.original_tflocoformer = TFLocoformerSeparator(
            input_dim=None,
            num_spk=1,
            n_layers=n_layers,
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,  # used only in RMSGroupNorm
            tf_order=tf_order,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,  # available when using mixed precision
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            # others
            eps=eps,
        )
        self.l1_loss = nn.L1Loss()
        fft_sizes = [256, 512, 768, 1024]
        hop_sizes = [n // 2 for n in fft_sizes]

        self.multiresolution_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=fft_sizes
        )

    def forward(self, y):
        y_scaled = y / y.std(dim=(-1, -2), keepdims=True)
        Y = self.stft_module(y_scaled)
        # B C F T -> B T F
        Y = Y[..., 0, :, :].transpose(-1, -2)
        ilens = None
        [S_hat], _, _ = self.original_tflocoformer(Y, ilens=None)
        # we need to pass along the normalization factor for the model to work
        return S_hat[:, None, :, :].transpose(-2, -1)

    def internal_loss(self, pred, s):
        S_hat = pred
        s_scaled = s / s.std(dim=(-1, -2), keepdims=True)
        S = self.stft_module(s_scaled)
        s_hat = self.get_time(pred, length=s.size(-1))
        loss_time = self.l1_loss(s_hat, s)
        loss_multiresolution = self.multiresolution_loss(s_hat, s)
        return loss_time + loss_multiresolution

    def get_stft(self, pred, **kwargs):
        pred_stft = pred
        return pred_stft

    def get_time(self, pred, length=None):
        pred_stft = pred
        return self.istft_module(pred_stft, length=length)


if __name__ == "__main__":
    dtype = torch.float16
    batch = torch.randn(1, 1, 16000, device="cuda", dtype=dtype), torch.randn(1, 1, 16000, device="cuda", dtype=dtype)
    model = TFLocoformer(
        # attention_dim=128,
        attention_dim=32,
        n_layers=4,  # B
        emb_dim=96,  # D
        norm_type="rmsgroupnorm",
        num_groups=4,
        tf_order="ft",
        n_heads=4,
        flash_attention=True,
        ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
        ffn_hidden_dim=[128, 128],
        conv1d_kernel=8,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-05,
    )
    model = model.to(device="cuda", dtype=dtype)
    s_hat = model.forward(batch[0])
    # res=model.training_step(batch, batch_idx=0)
