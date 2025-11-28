#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:07:47 2024

@author: louis
"""

import math
import torch
from torch import nn
import pyroomacoustics as pra

SPEED_OF_SOUND = pra.parameters.Physics().get_sound_speed()


def shoebox_surface(dim_1, dim_2, dim_3):
    return 2 * (dim_1 * dim_2 + dim_2 * dim_3 + dim_1 * dim_3)


class EarlyEchoesModel(nn.Module):
    def __init__(
        self,
        rir_length: int = 16383,
        fs: int = 16000,
    ):
        super().__init__()
        self.fs = fs
        self.rir_length = rir_length
        self.register_buffer("T", torch.arange(self.rir_length))

    def compute_end_index(self, rir_properties):
        raise NotImplementedError()

    def compute_early_mask(self, rir_properties):
        end_index = self.compute_end_index(rir_properties)
        if isinstance(end_index, float):
            return self.T < end_index
        return self.T < end_index.unsqueeze(-1)

    def rir_to_early(self, rir, rir_properties):
        early_mask = self.compute_early_mask(rir_properties)
        if early_mask.ndim == 1:
            early_mask = early_mask[(None,) * (rir.ndim - 1)].expand(*rir.shape[:-1], -1)
        rir[~early_mask] = 0.0
        return rir

    def rir_to_late(self, rir, rir_properties, include_direct_path: bool = True):
        # assert (rir[..., 0] != 0).all()
        early_mask = self.compute_early_mask(rir_properties)
        if early_mask.ndim == 1:
            early_mask = early_mask[(None,) * (rir.ndim - 1)].expand(*rir.shape[:-1], -1)
        if include_direct_path:
            early_mask = early_mask.contiguous()
            early_mask[..., 0] = False
        rir[early_mask] = 0.0
        return rir


class FixedTimeEarlyEnd(EarlyEchoesModel):
    def __init__(
        self,
        end_time: float = 0.020,
        rir_length: int = 16383,
        fs: int = 16000,
    ):
        super().__init__(rir_length=rir_length, fs=fs)
        self.end_time = end_time

    def compute_end_index(self, rir_properties=None):
        return self.end_time * self.fs


class MeanFreePathEarlyEnd(EarlyEchoesModel):
    """https://aes2.org/publications/elibrary-page/?id=10176"""

    def __init__(
        self,
        mean_free_path_multiplier: float = 2.0,  # Shuld be 3 but 2 to account for the direct path
        rir_length: int = 16383,
        fs: int = 16000,
    ):
        super().__init__(rir_length=rir_length, fs=fs)
        self.mean_free_path_multiplier = mean_free_path_multiplier

    def compute_end_index(self, rir_properties):
        surface = shoebox_surface(
            rir_properties["shoebox_width"], rir_properties["shoebox_height"], rir_properties["shoebox_length"]
        )
        mean_free_path = 4 * rir_properties["volume"] / surface
        mean_free_path_in_samples = mean_free_path / SPEED_OF_SOUND * self.fs
        late_reverb_begin = self.mean_free_path_multiplier * mean_free_path_in_samples
        return late_reverb_begin


class EchoesDensityReverbEnd(EarlyEchoesModel):
    """
    http://www.conforg.fr/acoustics2008/cdrom/data/articles/001386.pdf
    and Polack's PhD
    """

    def __init__(
        self,
        target_echoes_density: float = 10 / 0.024,  # Polack
        rir_length: int = 16383,
        fs: int = 16000,
    ):
        super().__init__(rir_length=rir_length, fs=fs)
        self.target_echoes_density = target_echoes_density

    def compute_end_index(self, rir_properties):
        begin_time = torch.sqrt(
            self.target_echoes_density * rir_properties["volume"] / (4 * torch.pi * SPEED_OF_SOUND**3)
        )
        return begin_time * self.fs


def compare_approaches():
    from datasets import SynthethicRirDataset, WSJSimulatedRirDataModule
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    cmap = list(mcolors.TABLEAU_COLORS.values())

    early_reverb_models = {
        "Fixed at 50ms": FixedTimeEarlyEnd(end_time=0.050),
        "Using echo density (target density of 10 echoes/24ms)": EchoesDensityReverbEnd(
            target_echoes_density=10 / 0.024
        ),
        "Using 2 times Mean free path after peak": MeanFreePathEarlyEnd(mean_free_path_multiplier=2.0),
    }

    plt.close("all")

    for query in [
        "rt_60<0.3",
        "0.5<rt_60<.6",
        "0.8<rt_60",
    ]:
        rir_dataset = SynthethicRirDataset(
            rir_root="./data/rirs_v2",
            query=query,
        )

        idx = np.random.randint(len(rir_dataset))
        rir, rir_properties = rir_dataset[idx]

        rir = rir[..., rir.abs().argmax() :]
        rir = rir / rir[..., 0]

        plt.figure()
        plt.suptitle(query)
        plt.plot(rir.squeeze(), label="original RIR")
        for i, (k, v) in enumerate(early_reverb_models.items()):
            plt.vlines(v.compute_end_index(rir_properties), ymin=0, ymax=1, label=k, colors=cmap[i + 1])
        plt.legend()
        plt.show()


def compute_mean_end_index_over_dataset(
    rir_root="./data/rirs_v2", model=MeanFreePathEarlyEnd(mean_free_path_multiplier=2.0)
):
    import numpy as np
    from datasets import SynthethicRirDataset
    import tqdm.auto as tqdm
    import matplotlib.pyplot as plt

    rir_dataset = SynthethicRirDataset(rir_root)
    res = np.zeros(len(rir_dataset))
    for i, (h, d) in enumerate(tqdm.tqdm(rir_dataset)):
        early_idx = model.compute_end_index(d)
        res[i] = early_idx
    print("mean", res.mean())
    print("median", np.median(res))
    print("std", np.std(res))
    plt.figure()
    plt.hist(res)
    return res


if __name__ == "__main__":
    compute_mean_end_index_over_dataset()
