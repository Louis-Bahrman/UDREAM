#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:59:57 2024

@author: louis
"""

import torch
from model.utils.abs_models import AbsReverbModel


class Dirac(AbsReverbModel):
    def __init__(self, rir_length: int = 16383):
        super().__init__()
        self.register_buffer("res", torch.zeros((1, rir_length)))
        self.res[..., 0] = 1

    def forward(self, input, *args, **kwargs):
        return self.res.expand(input.size(0), -1, -1)

    def internal_loss(self, pred, target):
        return 0
