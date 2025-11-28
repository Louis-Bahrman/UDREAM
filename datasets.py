#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains all utilities for dataset management

"""

import math
import copy
# import pyroomacoustics as pra
import numpy as np
import itertools
import torch
from torchaudio.datasets import LIBRISPEECH
import torchaudio.transforms
import lightning as L

import tqdm.auto as tqdm
import os
import pandas as pd
import soundfile as sf
import glob
# import stempeg

from torch.utils.data import random_split
from model.utils.tensor_ops import (
    crop_or_zero_pad_to_target_len,
    energy_to_db,
    db_to_amplitude,
    zero_pad,
)

# %% utils


def limit_dataset_size(
    dataset: torch.utils.data.Dataset, limit_size: float | int = 1.0
):
    """
    Create a subset of the `dataset` of size `limit_size`.

    Behaviour similar to L.Trainer.overfit_batches but used only for training sets
    (unlike `L.Trainer.overfit_batches` which applies a similar split to validation and test sets)

    This also solves the problem of `l.Trainer.limit_train_batches` which does not create a deterministic subset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Any dataset, such as EarsReverbDataset
    limit_size : float | int, optional:
        - If >1: number of samples of the dataset
        - If <1: uses a fraction of the dataset
        - The default is 1.0, which returns the dataset itself

    Raises
    ------
    ValueError
        if `limit_size` > `len(dataset)`.

    Returns
    -------
    torch.utils.data.Dataset
        Subset of target size.

    """
    if not isinstance(limit_size, (int, float)):
        raise ValueError("Only float or int limit_sizes are supported")
    if limit_size == 1.0 or limit_size == len(dataset):
        return dataset
    if isinstance(limit_size, float) and limit_size < 1.0:
        num_selected_indices = round(len(dataset) * limit_size)
    else:
        limit_size = int(limit_size)
        if limit_size > len(dataset):
            raise ValueError(
                "limit_size cannot be greater than the len of the dataset"
            )
        num_selected_indices = limit_size
    print(
        20 * "-"
        + "\n"
        + f"Using {num_selected_indices}/{len(dataset)} of the dataset\n"
        + 20 * "-"
    )
    indices = range(num_selected_indices)
    return torch.utils.data.Subset(dataset, indices)


# %% Synthethic RIR
class SynthethicRirDataset(torch.utils.data.Dataset):
    """Dataset of RIRs synthesized using the Image Source Method (ISM) from PyroomAcoustics."""

    def __init__(
        self,
        rir_root: str = "./data/rirs_v2",  # root of the RIR folder
        num_new_rooms: int = 0,  # number of new rooms to generate
        room_dim_range: tuple[float, float] = (
            5.0,
            10.0,
        ),  # width and length range of the rooms in meters
        room_height_range: tuple[float, float] = (
            2.5,
            4.0,
        ),  # height range of the room in meters
        rt60_range: tuple[float, float] = (0.2, 1.0),  # range of RT_60
        num_sources_per_room: int = 1,  # number of sources. If source_mic_distance is defined, force this to 1.
        num_mics_per_room: int = 16,  # number of mics in the room
        min_distance_to_wall: float = 0.5,  # min distance to wall to stay in the limits of ISM method
        mic_height_range: tuple[float, float] = (
            0.7,
            2,
        ),  # also used for source placement
        fs: int = int(16e3),  # Sample rate of the generated RIR
        query: str = "",  # Used in pandas.Dataframe.filter
        return_properties: list[str] | None = [
            "volume",
            "shoebox_length",
            "shoebox_width",
            "shoebox_height",
            "rt_60",
        ],  # properties to return at sampling
        source_mic_distance_range: tuple[float, float] | None = (
            0.75,
            2.5,
        ),  # None for no constraints on distance
    ):
        """
        Initialize a Synthethic RIR dataset.

        RIRs are sampled using the Image source method (ISM) using pyroomacoustics.

        If `num_new_rooms > 1`, also generates rooms, else only samples from existing ones.

        Parameters
        ----------
        rir_root : str, optional
            The root directory where the RIR data is stored. Defaults to "./data/rirs_v2".

        num_new_rooms : int, optional
            The number of new rooms to generate. Defaults to 0.

        room_dim_range : tuple of float, optional
            A tuple specifying the minimum and maximum width and length (in meters) of the rooms.
            Defaults to (5.0, 10.0).

        room_height_range : tuple of float, optional
            A tuple specifying the minimum and maximum height (in meters) of the room. Defaults to (2.5, 4.0).

        rt60_range : tuple of float, optional
            A tuple specifying the range of RT60 (reverberation time) values for the rooms.
            Defaults to (0.2, 1.0).

        num_sources_per_room : int, optional
            The number of sound sources per room. If the `source_mic_distance_range` is defined, this value is forced to 1. Defaults to 1.

        num_mics_per_room : int, optional
            The number of microphones per room. Defaults to 16.

        min_distance_to_wall : float, optional
            The minimum distance to the wall to stay within the limits of the ISM method. Defaults to 0.5.

        mic_height_range : tuple of float, optional
            A tuple specifying the minimum and maximum height (in meters) for microphone placement.
            Also used for the placement of sources. Defaults to (0.7, 2.0).

        fs : int, optional
            The sample rate (in Hz) of the generated RIRs. Defaults to 16kHz (16000 Hz).

        query : str, optional
            A query string used for filtering data within a pandas DataFrame.
            Example: `query="rt_60 > 0.8"` returns a subset of the dataset with a rt_60 > 0.8.
            Defaults to an empty string.

        return_properties : list of str or None, optional
            A list of properties to return during the sampling process, such as room volume, shoebox dimensions,
            and reverberation time (RT60). If set to None, no properties will be returned. Defaults to:
            ["volume", "shoebox_length", "shoebox_width", "shoebox_height", "rt_60"].

        source_mic_distance_range : tuple of float or None, optional
            A tuple specifying the minimum and maximum distance (in meters) between the sound source and the microphones.
            If None, there are no constraints on the distance. Defaults to (0.75, 2.5).

        """
        # parse data to construct path
        self.rir_root = rir_root
        self.num_new_rooms = num_new_rooms
        self.room_dim_range = room_dim_range
        self.room_height_range = room_height_range
        self.rt60_range = rt60_range
        self.num_sources_per_room = num_sources_per_room
        self.num_mics_per_room = num_mics_per_room
        self.min_distance_to_wall = min_distance_to_wall
        self.mic_height_range = mic_height_range
        self.fs = fs
        self.query = query
        self.return_properties = return_properties or []
        self.source_mic_distance_range = source_mic_distance_range

        self.rir_properties = None
        self.filtered_rir_properties = None

        if self.num_new_rooms == 0:
            self._read_rir_csv()
            self._filter_rir_properties()

        if self.source_mic_distance_range is not None:
            # we force new behaviour
            self.num_mics_per_room = (
                self.num_sources_per_room * self.num_mics_per_room
            )
            self.num_sources_per_room = 1

    @property
    def rir_csv_path(self):
        return os.path.join(self.rir_root, "properties.csv")

    @property
    def num_filtered_rooms(self):
        return len(self.filtered_rir_properties["room_idx"].unique())

    @property
    def num_total_rooms(self):
        return len(self.rir_properties["room_idx"].unique())

    @property
    def num_filtered_rirs(self):
        return len(self.filtered_rir_properties.index)

    @property
    def num_rirs(self):
        return len(self.rir_properties.index)

    def _room_path(self, room_idx):
        return os.path.join(self.rir_root, f"room_{room_idx}")

    def _rir_path(self, room_idx, rir_idx_in_room):
        return os.path.join(
            self._room_path(room_idx), f"rir_{rir_idx_in_room}.wav"
        )

    def _read_rir_csv(self):
        self.rir_properties = pd.read_csv(
            self.rir_csv_path, index_col="rir_global_idx"
        )

    def _write_rir_csv(self):
        self.rir_properties.to_csv(self.rir_csv_path, float_format="%.3f")

    def _filter_rir_properties(self):
        if self.query is not None and self.query != "":
            self.filtered_rir_properties = self.rir_properties.query(
                self.query
            )
            # if len(self.filtered_rir_properties) == 0:
            #     raise ValueError("filtering returned empty dataset, try widening the search")
        else:
            self.filtered_rir_properties = self.rir_properties

    def generate_data_if_needed(self):
        if self.num_new_rooms > 0:
            self._generate_data()

    def _valid_position_range(self, shoebox_dim):
        return (
            [
                self.min_distance_to_wall,
                self.min_distance_to_wall,
                self.mic_height_range[0],
            ],
            [
                *(shoebox_dim - self.min_distance_to_wall)[:2],
                self.mic_height_range[1],
            ],
        )
        # return np.vstack((np.vstack((2 * [self.min_distance_to_wall], shoebox_dim[:2])).T, self.mic_height_range))
        #     zip(2 * [self.min_distance_to_wall], shoebox_dim[:2] - self.min_distance_to_wall, self.mic_height_range)
        # )

    def _sample_uniform_positions(self, shoebox_dim, num_positions=1):
        return self.rng.uniform(
            *self._valid_position_range(shoebox_dim), size=(num_positions, 3)
        )

    def _is_valid_position(self, position, shoebox_dim):
        eps = 1e-5
        return (
            (self.min_distance_to_wall - eps <= position).all()
            and (
                position <= shoebox_dim - self.min_distance_to_wall + eps
            ).all()
            and (self.mic_height_range[0] - eps <= position[..., -1]).all()
            and (position[..., -1] <= self.mic_height_range[1] + eps).all()
        )

    def _generate_data(self):
        import pyroomacoustics as pra
        self.rng = np.random.default_rng()
        os.makedirs(self.rir_root, exist_ok=True)
        if not os.path.isfile(self.rir_csv_path):
            # generate file

            self.rir_properties = pd.DataFrame(
                columns=[
                    "rir_global_idx",
                    "rir_path",
                    # Room properties
                    "room_idx",
                    "shoebox_length",
                    "shoebox_width",
                    "shoebox_height",
                    "volume",
                    "rt_60",
                    "absorption",
                    # rir_properties
                    "rir_idx_in_room",
                    "source_idx",
                    "mic_idx",
                    "source_x",
                    "source_y",
                    "source_z",
                    "mic_x",
                    "mic_y",
                    "mic_z",
                    "source_mic_distance",
                ]
            ).set_index("rir_global_idx")
            self._write_rir_csv()
            # self.rooms_properties=pd.DataFrame(columns=["room_idx","room_path", "shoebox_dim", "volume", "rt_60", "absorption"]).set_index("room_idx")
            # self._write_rooms_csv()
        # self._read_rooms_csv()
        self._read_rir_csv()
        current_num_new_rooms = 0
        rir_global_idx = (
            self.rir_properties.index.max() + 1 if self.num_rirs > 0 else 1
        )

        room_idx = (
            self.rir_properties["room_idx"].max() + 1
            if self.num_rirs > 0
            else 1
        )

        with tqdm.tqdm(total=self.num_new_rooms) as pbar:
            while current_num_new_rooms < self.num_new_rooms:
                shoebox_dim = np.zeros(3)
                shoebox_dim[:2] = self.rng.uniform(*self.room_dim_range, 2)
                shoebox_dim[2] = self.rng.uniform(*self.room_height_range)
                rt60 = self.rng.uniform(*self.rt60_range)
                try:
                    absorption, max_order = pra.inverse_sabine(
                        rt60, shoebox_dim
                    )
                except ValueError:
                    # Room too large for rt60
                    pass
                else:
                    room = pra.ShoeBox(
                        shoebox_dim,
                        self.fs,
                        absorption=absorption,
                        max_order=max_order,
                        use_rand_ism=True,
                    )
                    volume = np.prod(room.shoebox_dim)
                    room_path = self._room_path(room_idx)
                    assert not os.path.isdir(
                        room_path
                    ), "Error in room_idx generation"
                    # add sources
                    source_pos = self._sample_uniform_positions(
                        shoebox_dim, num_positions=self.num_sources_per_room
                    )
                    assert self._is_valid_position(source_pos, shoebox_dim)
                    for sp in source_pos:
                        room.add_source(sp)

                    # add mic old behaviour: Uniform sampling
                    if self.source_mic_distance_range is None:
                        # old behiaviour
                        mic_pos = self._sample_uniform_positions(
                            shoebox_dim, num_positions=self.num_mics_per_room
                        )
                        assert self._is_valid_position(mic_pos, shoebox_dim)
                        room.add_microphone_array(mic_pos.T)
                    # new behaviour: Sampling of position with rejection if the distance is not within range
                    else:
                        mic_idx = 0
                        while mic_idx < self.num_mics_per_room:
                            mic_pos_within_room = False
                            source_mic_distance = self.rng.uniform(
                                *self.source_mic_distance_range
                            )
                            while not mic_pos_within_room:
                                normal_3d = self.rng.normal(size=3)
                                source_mic_vector = (
                                    source_mic_distance
                                    * normal_3d
                                    / np.linalg.norm(normal_3d)
                                )
                                assert np.allclose(
                                    np.linalg.norm(source_mic_vector),
                                    source_mic_distance,
                                )
                                mic_pos = source_pos + source_mic_vector
                                mic_pos_within_room = self._is_valid_position(
                                    mic_pos, shoebox_dim
                                )
                            mic_idx += 1
                            room.add_microphone(mic_pos.T)

                    room.compute_rir()

                    os.mkdir(room_path)
                    for rir_idx_in_room, (source_idx, mic_idx) in enumerate(
                        itertools.product(
                            range(self.num_sources_per_room),
                            range(self.num_mics_per_room),
                        )
                    ):
                        source_pos = room.sources[source_idx].position
                        mic_pos = room.mic_array.R.T[mic_idx]
                        source_mic_distance = np.linalg.norm(
                            source_pos - mic_pos
                        )
                        rir_path = self._rir_path(room_idx, rir_idx_in_room)
                        rir_properties_dict = {
                            "rir_path": rir_path,
                            # Room properties
                            "room_idx": room_idx,
                            "shoebox_length": shoebox_dim[0],
                            "shoebox_width": shoebox_dim[1],
                            "shoebox_height": shoebox_dim[2],
                            "volume": volume,
                            "rt_60": rt60,
                            "absorption": absorption,
                            # rir_properties
                            "rir_idx_in_room": rir_idx_in_room,
                            "source_idx": source_idx,
                            "mic_idx": mic_idx,
                            "source_x": source_pos[0],
                            "source_y": source_pos[1],
                            "source_z": source_pos[2],
                            "mic_x": mic_pos[0],
                            "mic_y": mic_pos[1],
                            "mic_z": mic_pos[2],
                            "source_mic_distance": source_mic_distance,
                        }

                        self.rir_properties.loc[rir_global_idx] = (
                            rir_properties_dict
                        )
                        rir = room.rir[mic_idx][source_idx]
                        sf.write(rir_path, rir, self.fs)
                        rir_global_idx += 1

                    room_idx += 1
                    current_num_new_rooms += 1
                    self._write_rir_csv()
                    pbar.update(1)

        self._write_rir_csv()
        self._filter_rir_properties()

    def __len__(self):
        return self.num_filtered_rirs

    def __getitem__(self, idx):
        """
        Return RIR and rir_properties
        -------
        waveform : torch.Tensor
            Shape [1,n] where n is the length (in samples of the RIR).
        other_properties : Dict
            Dict containing the rir_properties.

        """
        # Use iloc
        rir_row = self.filtered_rir_properties.iloc[idx]
        rir_path = rir_row["rir_path"]
        waveform, sample_rate = torchaudio.load(rir_path)
        if sample_rate != self.fs:
            raise ValueError(
                f"sample rate should be {self.fs}, but got {sample_rate}"
            )
        other_properties = {
            k: torch.tensor(v).unsqueeze(0)
            for k, v in rir_row[self.return_properties].items()
        }
        return waveform, other_properties

    def random_split_by_rooms(self, *proportions):
        """
        Perform random splitting.

        Unlike `torch.utils.data.random_split`,
        ensures that RIRs from the same room are in the same subset

        Parameters
        ----------
        *proportions : float
            proportions of val and test (<1.).
            The proportion of train is automatically computed

        Returns
        -------
        subsets : torch.utils.data.Dataset
            train, val, and test subsets.

        """
        # Train test splits is implemented here since we use room information
        unique_room_idxs = self.filtered_rir_properties["room_idx"].unique()

        # We use torch random split function which is easier and also works with integers
        proportions = (1 - sum(proportions), *proportions)
        rooms_of_each_subset = random_split(unique_room_idxs, proportions)

        subsets = []
        # Only a shallow copy is needed, we will only modify the query method, not the dataframe
        for rooms_of_subset in rooms_of_each_subset:
            subset = copy.copy(self)
            subset.add_filter_to_query(
                f"room_idx.isin({list(rooms_of_subset)})"
            )
            subset._filter_rir_properties()
            subsets.append(subset)
        return subsets

    def add_filter_to_query(self, filter_to_add):
        if self.query == "":
            self.query = filter_to_add
        else:
            self.query += " & " + filter_to_add


# %% RIR transforms


class ConvolveDryWithEarly(torch.nn.Module):
    """Convolves a dry signal with early reverberation."""

    def __init__(self, early_echoes_masking_module: torch.nn.Module):
        super().__init__()
        self.early_echoes_masking_module = early_echoes_masking_module
        self.convolution_transform = torchaudio.transforms.FFTConvolve(
            mode="full"
        )

    def forward(self, x, h, rir_properties):
        h_early = self.early_echoes_masking_module.rir_to_early(
            h[None, ...].clone(), rir_properties
        )[0, ...]
        return self.convolution_transform(x, h_early)[..., : x.size(-1)]


class RandomModifyDRR(torch.nn.Module):
    """
    Applies a gain uniformly sampled within a dB range on the first n samples of the rir.

    This causes the direct to reverberant ratio (DRR) to change.
    """

    def __init__(
        self, min_db_gain=-12, max_db_gain=3, apply_on_first_n_samples=80
    ):
        super().__init__()
        self.min_db_gain = min_db_gain
        self.max_db_gain = max_db_gain
        self.apply_on_first_n_samples = apply_on_first_n_samples

    def forward(self, h, *args):
        assert torch.allclose(h[..., 0].abs(), torch.ones_like(h[..., 0]))
        gain_multiplier_db = (
            self.max_db_gain - self.min_db_gain
        ) * torch.rand(h.shape[:-1]) + self.min_db_gain
        orig_energy_h_db = energy_to_db(
            h[..., : self.apply_on_first_n_samples]
            .abs()
            .square()
            .sum(dim=-1, keepdim=True)
        )
        gain_db = gain_multiplier_db.unsqueeze(-1)  # + orig_energy_h_db
        gain_linear = db_to_amplitude(gain_db)
        h[..., : self.apply_on_first_n_samples] = (
            gain_linear * h[..., : self.apply_on_first_n_samples]
        )
        return h


class NormalizeEnergy(torch.nn.Module):
    """RMS normalization of the RIR."""

    def forward(self, h):
        h_energy = h.abs().square().sum(dim=-1, keepdim=True)
        return h / h_energy.sqrt()


class DARRirAugmentation(torch.nn.Module):
    """
    RandomModyfyDRR and NormalizeEnergy augmentations.

    Augmentations used in `Differentiable Artificial Reverberation <https://doi.org/10.1109/TASLP.2022.3193298>`_.

    """

    def __init__(self):
        super().__init__()
        self.submodules = torch.nn.Sequential(
            RandomModifyDRR(), NormalizeEnergy()
        )

    def forward(self, h, *args):
        return self.submodules(h)


class RIRToLate(torch.nn.Module):
    """
    Returns the late reverberation only of a rir.

    The direct path should be at the first sample of the RIR

    """

    def __init__(
        self,
        early_echoes_masking_module: torch.nn.Module,
        include_direct_path: bool = True,  # whether to include the peak of the RIR
    ):
        super().__init__()
        self.early_echoes_masking_module = early_echoes_masking_module
        self.include_direct_path = include_direct_path

    def forward(self, h, rir_properties):
        return self.early_echoes_masking_module.rir_to_late(
            h.clone()[None, ...],
            rir_properties,
            include_direct_path=self.include_direct_path,
        )[0, ...]


# %% Dry and wet datasets and datamodules


def normalize_sox(x: torch.Tensor, sample_rate: int = 16000):
    # dependant on samplerate so should be avoided
    return torchaudio.sox_effects.apply_effects_tensor(
        x, sample_rate=sample_rate, effects=[["norm"]]
    )[0]


def normalize_max(x: torch.Tensor, target_max: float = 0.5):
    # 0.5 instead of sth ike 0.98 in order to not saturate when convolving with k
    return x / x.abs().max() * target_max


def remove_silent_windows(
    x: torch.Tensor, silence_power: float = -20.0, window_len: int = 1024
):
    x_split = x.split(window_len, dim=-1)
    x_split_nonsilent = [
        t for t in x_split if energy_to_db(t.norm() ** 2) > silence_power
    ]
    return torch.cat(x_split_nonsilent, dim=-1)


class AudioDatasetConvolvedWithRirDataset(torch.utils.data.Dataset):
    """
    For each audio signal, picks a random rir and convolve.

    return format (wet, (dry, rir, rir_properties))

    """

    def __init__(
        self,
        audio_dataset,
        rir_dataset,
        dry_signal_target_len: int | None = 32767,
        rir_target_len: int | None = 16383,
        align_and_scale_to_direct_path=True,
        dry_signal_start_index=16000,  # None for random start
        pre_associate=False,
        convolve_here: bool = True,  # else convolve afterwards on GPU
        resampling_transform: torch.nn.Module | None = None,
        normalize: bool = True,
        ignore_silent_windows: bool = True,
        rir_transforms_for_wet: (
            torch.nn.Module | None
        ) = None,  # Applied on both h and y
        dry_only_transforms: (
            torch.nn.Module | None
        ) = None,  # Applied on s only, not on y
        rir_only_transforms: (
            torch.nn.Module | None
        ) = None,  # applied on h only (not on y or s)
        index_according_to_rir_dataset: bool = False,
        enable_caching: bool = True,
        limit_size: int | float = 1.0,
    ):
        """
        Instantiate a dataset of dry audios convolved with RIRs.

        Parameters
        ----------
        audio_dataset : torch.utils.data.Dataset
            Dataset of dry audios.
        rir_dataset : torch.utils.data.Dataset
            Dataset of RIRs.
        dry_signal_target_len : int | None, optional
            If set, dry signal is cropped or zero-padded to this length. The default is 32767.
        rir_target_len : int | None, optional
            RIR is cropped or zero-padded to this length. The default is 16383.
        align_and_scale_to_direct_path : bool, optional
            Whether to start the RIR at the direct path and set its amplitude to 1.

            The default is True.
        dry_signal_start_index : int, optional
            First sample of the dry signal returned.
            If not set, randomly selected in `(0, len(dry_signal)-dry_signal_target_len)`.
            The default is 16000.
        pre_associate : bool, optional
            Whether to use dynamic mixing

            - If true: no dynamic mixing, each audio will be convolved deterministically with the same RIR at each time it is sampled
            - If false: dynamic mixing, every time an audio is sampled, it will be convolved with a randomly-selected RIR

            The default is False.
        convolve_here : bool, optional
            Whether the convolution between dry and wet should be performed on the CPU or GPU.

            - if True: convolution performed on CPU

            - If False: convolution is delayed to the GPU, in the method `on_after_batch_transfer`
            The wet output of __getitem__ will be a tensor full of `nan`

            The default is True.
        resampling_transform : torchaudio.transforms.Resample | None, optional
            Resampling transform applied on both RIR and dry speech before convolving them.
            The default is None.
        normalize : bool, optional
            Whether to normalize the dry signal before convolution. The default is True.
        ignore_silent_windows : bool, optional
            Whether to remove silences in the dry signal before convolving it with the RIR.
            If True, splits the dry audio in 1024-samples long segments.
            For each segment, checks whether its total energy is greater than 20 dB.
            If not, the segment is discarded.
            The default is True.
        rir_transforms_for_wet : torch.nn.Module | None, optional
            RIR transforms only applied on wet signal, not on the dry signal.
            The default is None.

            Warning, not thoroughly tested
        dry_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the dry signal only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        rir_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the RIR only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        index_according_to_rir_dataset : bool, optional
            Whether to index the dataset by the dry signal or the RIR index.

            - if True: Indexes according to the RIR dataset.
            The length of the convolved dataset will be the length of the RIR dataset.

            - if False: Indexes according to the dry audios dataset.
            The length of the convolved dataset will be the length of `audio_dataset`.

            The default is False.
        enable_caching : bool, optional
            Whether to cache audios and RIRs in the CPU RAM for faster access.
            If so, `__init__` might take some time.
            The default is True.
        limit_size : int | float, optional
            Limit of the size of both RIR and dry audio datasets.
            See `limit_dataset_size`.
            The default is 1.0.

        Raises
        ------
        NotImplementedError
            If one of `rir_transforms_for_wet`, `dry_only_transforms`, or `rir_only_transforms`
            is set and `convolve_here is False` or.the returned RIR properties are not empty.
        """
        self.limit_size = limit_size
        self.audio_dataset = limit_dataset_size(
            audio_dataset, limit_size=self.limit_size
        )
        self.rir_dataset = limit_dataset_size(
            rir_dataset, limit_size=self.limit_size
        )
        self.pre_associate = pre_associate and len(self.rir_dataset) > 0
        if self.pre_associate:
            self.pre_association = torch.randint(
                len(self.rir_dataset), size=(len(self.audio_dataset),)
            )
        self.dry_signal_target_len = dry_signal_target_len
        self.rir_target_len = rir_target_len
        self.align_and_scale_to_direct_path = align_and_scale_to_direct_path
        self.dry_signal_start_index = dry_signal_start_index
        self.convolve_here = convolve_here
        self.resampling_transform = resampling_transform
        self.normalize = normalize
        self.normalize_op = normalize_max
        self.ignore_silent_windows = ignore_silent_windows
        self.rir_transforms_for_wet = rir_transforms_for_wet
        self.dry_only_transforms = dry_only_transforms
        self.rir_only_transforms = rir_only_transforms
        self.index_according_to_rir_dataset = index_according_to_rir_dataset
        if index_according_to_rir_dataset and self.pre_associate:
            self.pre_association = torch.randint(
                len(self.audio_dataset), size=(len(self.rir_dataset),)
            )

        if (
            self.dry_only_transforms is not None
            or self.rir_only_transforms is not None
        ) and not self.convolve_here:
            raise NotImplementedError()

        assert (
            self.normalize or not self.ignore_silent_windows
        ), "need to normalize in order to ignore_silent_windows"

        if self.convolve_here:
            self.convolution_transform = torchaudio.transforms.FFTConvolve(
                mode="full"
            )

        self.enable_caching = enable_caching
        if self.enable_caching and (
            self.rir_transforms_for_wet is not None
            or self.rir_only_transforms is not None
            or self.rir_target_len is None
            or not (
                self.rir_dataset.return_properties is None
                or len(self.rir_dataset.return_properties) == 0
            )
        ):
            raise NotImplementedError()
        if self.enable_caching:
            self.rir_cache = torch.full(
                (len(rir_dataset), 1, self.rir_target_len), torch.nan
            )
            self.dry_cache = dict()
            self._cache_all_audios()

    def __len__(self):
        if self.index_according_to_rir_dataset:
            return len(self.rir_dataset)
        return len(self.audio_dataset)

    def get_rir(self, rir_idx):
        try:
            if not self.enable_caching:
                raise KeyError()
            rir_aligned_scaled_cropped = self.rir_cache[rir_idx]
            if rir_aligned_scaled_cropped.isnan().any():
                raise KeyError()
            rir_properties = dict()
        except KeyError:
            rir, rir_properties = self.rir_dataset[rir_idx]
            if self.resampling_transform:
                rir = self.resampling_transform(rir)
            # Align and scale dry and RIR to direct path
            if self.align_and_scale_to_direct_path:
                peak_index = torch.argmax(torch.abs(rir))
                rir_peak = rir[..., peak_index]
                rir_aligned = rir[..., peak_index:]
                rir_aligned_scaled = rir_aligned / rir_peak
                if "rt_60" in rir_properties.keys():
                    # raise NotImplementedError(
                    #     "I don't know if it is really needed"
                    # )
                    rir_properties["rt_60"] -= peak_index / self.rir_dataset.fs
            else:
                rir_aligned_scaled = rir
            if self.rir_target_len is not None:
                rir_aligned_scaled_cropped = crop_or_zero_pad_to_target_len(
                    rir_aligned_scaled, target_len=self.rir_target_len
                )
            else:
                rir_aligned_scaled_cropped = rir_aligned_scaled
            if self.enable_caching:
                self.rir_cache[rir_idx] = rir_aligned_scaled_cropped
        return rir_aligned_scaled_cropped, rir_properties

    def get_dry(self, audio_idx):
        try:
            # if not use cache skip to default loading
            if not self.enable_caching:
                raise KeyError()
            x_full = self.dry_cache[audio_idx]
        except KeyError:
            x_full = self.audio_dataset[audio_idx]

            if self.normalize:
                x_full = self.normalize_op(x_full)
            if self.ignore_silent_windows:
                x_full = remove_silent_windows(x_full)

            if self.resampling_transform:
                x_full = self.resampling_transform(x_full)
            if self.enable_caching:
                # put in cache
                self.dry_cache[audio_idx] = x_full
        return x_full

    def __getitem__(self, idx):
        if self.index_according_to_rir_dataset:
            rir_idx = idx
            if self.pre_associate:
                audio_idx = int(self.pre_association[rir_idx])
            else:
                audio_idx = int(torch.randint(len(self.audio_dataset), (1,)))
        else:
            audio_idx = idx
            # Pick RIR
            if self.pre_associate:
                rir_idx = int(self.pre_association[audio_idx])
            else:
                rir_idx = int(torch.randint(len(self.rir_dataset), (1,)))

        # Pick dry, use cache if necessary
        x_full = self.get_dry(audio_idx)

        # get rir, use cache if necessary
        rir_aligned_scaled_cropped, rir_properties = self.get_rir(rir_idx)

        # get section of dry signal
        if self.dry_signal_start_index is None:
            start_index = torch.randint(
                max(1, x_full.shape[-1] - self.dry_signal_target_len),
                size=(1,),
            )[0]
        else:
            start_index = self.dry_signal_start_index
            if start_index >= x_full.size(-1):
                print("tensor not long enough to be cropped, skipping")
                return self.__getitem__(idx + 1)
        x = x_full[..., start_index:]
        # Crop for linear convolution
        if self.dry_signal_target_len is not None:
            x_cropped = crop_or_zero_pad_to_target_len(
                x, target_len=self.dry_signal_target_len
            )
        else:
            x_cropped = x
        x_cropped = x_cropped - x_cropped.mean()
        # Perform convolution on align rir and non scaled audio
        # We use x_cropped and not x_scaled_cropped for convolution because we don't want to scale 2 times
        if self.rir_transforms_for_wet is not None:
            rir_transformed_for_wet = self.rir_transforms_for_wet(
                rir_aligned_scaled_cropped.clone(), rir_properties
            )
        else:
            rir_transformed_for_wet = rir_aligned_scaled_cropped.clone()
        if self.convolve_here:
            y = self.convolution_transform(rir_transformed_for_wet, x_cropped)
        else:
            y = torch.full(
                x.shape[:-1]
                + (
                    x_cropped.shape[-1]
                    + rir_transformed_for_wet.shape[-1]
                    - 1,
                ),
                fill_value=torch.nan,
            )
        if self.dry_only_transforms is not None:
            x_transformed = self.dry_only_transforms(
                x_cropped, rir_aligned_scaled_cropped.clone(), rir_properties
            )
        else:
            x_transformed = x_cropped
        if self.rir_only_transforms is not None:
            rir_only_transformed = self.rir_only_transforms(
                rir_aligned_scaled_cropped.clone(), rir_properties
            )
        else:
            rir_only_transformed = rir_aligned_scaled_cropped.clone()
        return y, (x_transformed, rir_only_transformed, rir_properties)

    def _cache_all_audios(self):
        print("caching dry signals")
        for dry_idx in tqdm.trange(len(self.audio_dataset)):
            _ = self.get_dry(dry_idx)
        print("caching RIRs")
        for rir_idx in tqdm.trange(len(self.rir_dataset)):
            _ = self.get_rir(rir_idx)

    def export(self, base_path: str, fs=16000, crop: int | None = None):
        """
        Export the dataset to files

        Parameters
        ----------
        base_path :
            Path to export the dataset to.
        fs : TYPE, optional
            Sampling rate. The default is 16000.
        crop : int | None, optional
            whether to crop signals to a given length. The default is None.

        """
        os.makedirs(os.path.join(base_path, "wet"))
        os.makedirs(os.path.join(base_path, "dry"))
        os.makedirs(os.path.join(base_path, "rir"))
        properties = []
        for idx in tqdm.trange(len(self)):
            y, (s, h, rir_properties) = self.__getitem__(idx)
            # print(y.abs().max())
            scaling_factor = 1 / torch.maximum(y.abs().max(), s.abs().max())
            y = scaling_factor * y
            s = scaling_factor * s
            properties.append({k: v.item() for k, v in rir_properties.items()})
            if crop is not None:
                y = y[..., :crop]
                s = s[..., :crop]
            torchaudio.save(
                os.path.join(base_path, "wet", str(idx) + ".wav"),
                y,
                sample_rate=fs,
            )
            torchaudio.save(
                os.path.join(base_path, "dry", str(idx) + ".wav"),
                s,
                sample_rate=fs,
            )
            torchaudio.save(
                os.path.join(base_path, "rir", str(idx) + ".wav"),
                h,
                sample_rate=fs,
            )
        df = pd.DataFrame(properties)
        df.index.name = "idx"
        df.to_csv(
            os.path.join(base_path, "properties.csv"), float_format="%.3f"
        )


class AudioDatasetConvolvedWithRirDatasetDataModule(L.LightningDataModule):
    """
    Abstract class to combine audio dataset and RIR dataset.

    You should inherit from this class and define your own "split_audio_dataset" method
    """

    def __init__(
        self,
        rir_dataset: torch.utils.data.Dataset,  # Dataset used at train and val
        rir_dataset_test: (
            torch.utils.data.Dataset | None
        ) = None,  # for testing only
        batch_size: int = 8,
        audio_root: str = "./data/speech",
        dry_signal_target_len: int = 32767,
        rir_target_len: int = 16383,
        align_and_scale_to_direct_path: bool = True,
        dry_signal_start_index_train: int | None = None,
        dry_signal_start_index_val_test: int | None = 16000,
        proportion_val_audio: (
            float | None
        ) = 0.1,  # Switch behaviour to use ready-made val split or not
        proportion_val_rir: float = 0.1,
        num_workers: int = 8,
        convolve_on_gpu: bool = False,
        resampling_transform: torch.nn.Module | None = None,
        num_distinct_rirs_per_batch: int | None = None,
        normalize: bool = True,
        ignore_silent_windows: bool = True,
        rir_transforms_for_wet: torch.nn.Module | None = None,
        rir_only_transforms: torch.nn.Module | None = None,
        dry_only_transforms: torch.nn.Module | None = None,
        index_according_to_rir_dataset: bool = False,
        dynamic_mixing_at_training: bool = True,
        prefetch_factor: int = 2,
        enable_caching_train: bool = False,
        enable_caching_val: bool = False,
        limit_training_size: float | int = 1.0,
    ):
        """
        Instantiate `AudioDatasetConvolvedWithRirDatasetDataModule`.

        Parameters
        ----------
        rir_dataset :
            RIR dataset used at train and val
        rir_dataset_test :
            RIR dataset used for testing only
        batch_size :
            batch_size
        audio_root :
            Dry signal dataset root
        dry_signal_target_len : int | None, optional
           If set, dry signal is cropped or zero-padded to this length. The default is 32767.
        rir_target_len : int | None, optional
            RIR is cropped or zero-padded to this length. The default is 16383.
        align_and_scale_to_direct_path : bool, optional
            Whether to start the RIR at the direct path and set its amplitude to 1.
            The default is True.
        dry_signal_start_index_train : int, optional
            First sample of the dry signal returned.
            If not set, randomly selected in `[0,len(dry_signal)-dry_signal_target_len]`.
            The default is None.
        dry_signal_start_index_val_test : int, optional
            Dry signal start on val and test sets
            If not set, randomly selected in `(0, len(dry_signal)-dry_signal_target_len)`.
            The default is 16000.
        proportion_val_audio :
            Proportion of the dry audio training set used for validation
        proportion_val_rir :
            Proportion of the RIR training set used for validation
        num_workers : int
            See `torch.utils.data.DataLoader`
        convolve_on_gpu : bool, optional
            Whether the convolution between dry and wet should be performed on the CPU or GPU.

            - if False: convolution performed on CPU

            - If True: convolution is delayed to the GPU, in the method `on_after_batch_transfer`

            The default is True.
        resampling_transform : torchaudio.transforms.Resample | None, optional
            Resampling transform applied on both RIR and dry speech before convolving them.
            The default is None.
        num_distinct_rirs_per_batch :
            If set, several dry audios will be convolved with the same RIR inside a batch.
            See `on_before_batch_transfer` method.
            If None, defaults to regular dynamic mixing.
            Warning, not thoroughly tested.
            Defaults to None.
        normalize : bool, optional
            Whether to normalize the dry signal before convolution. The default is True.
        ignore_silent_windows : bool, optional
            Whether to remove silences in the dry signal before convolving it with the RIR.
            If True, splits the dry audio in 1024-samples long segments.
            For each segment, checks whether its total energy is greater than 20 dB.
            If not, the segment is discarded.
            The default is True.
        rir_transforms_for_wet : torch.nn.Module | None, optional
            RIR transforms only applied on wet signal, not on the dry signal.
            The default is None.

            Warning, not thoroughly tested
        dry_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the dry signal only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        rir_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the RIR only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        index_according_to_rir_dataset : bool, optional
            Whether to index the dataset by the dry signal or the RIR index.

            - if True: Indexes according to the RIR dataset.
            The length of the convolved dataset will be the length of the RIR dataset.

            - if False: Indexes according to the dry audios dataset.
            The length of the convolved dataset will be the length of `audio_dataset`.

            The default is False.
        dynamic_mixing_at_training : bool, optional
            Whether to use dynamic mixing at training. Defaults to True.
        prefetch_factor:
            See `torch.utils.data.DataLoader`
        enable_caching_train : bool, optional
            Whether to cache audios and RIRs from the training set in the CPU RAM for faster access.
            If so, `__init__` might take some time.
            The default is False.
        enable_caching_val : bool, optional
            Whether to cache audios and RIRs from the val set in the CPU RAM for faster access.
            If so, `__init__` might take some time.
            Caching is never enabled for test, since only one pass of the dataset is done.
            The default is False.
        limit_training_size : int | float, optional
            Limit of the size of both RIR and dry audio datasets at training.
            See `limit_dataset_size`.
            The default is 1.0.

        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "rir_dataset",
                "rir_dataset_test",
                "rir_further_transforms",
                "dry_only_transforms",
                "rir_only_transforms",
                "rir_transforms_for_wet",
            ]
        )
        self.save_hyperparameters(ignore=[], logger=False)

        self.rir_dataset = rir_dataset
        self.rir_dataset_test = rir_dataset_test
        if not os.path.isdir(self.hparams.audio_root):
            os.makedirs(self.hparams.audio_root)

        self.convolution_transform = torchaudio.transforms.FFTConvolve(
            mode="full"
        )

    def prepare_data(self):
        if hasattr(self.rir_dataset, "generate_data_if_needed"):
            self.rir_dataset.generate_data_if_needed()

    def split_audio_dataset(self, stage=None):
        raise NotImplementedError()

    def setup(self, stage=None):
        self.split_audio_dataset()
        assert (
            hasattr(self, "dry_train")
            and hasattr(self, "dry_val")
            and hasattr(self, "dry_test")
        )
        self.generate_convolved_datasets()

    def generate_convolved_datasets(self):
        self.rir_dataset_train, self.rir_dataset_val = (
            self.rir_dataset.random_split_by_rooms(
                self.hparams.proportion_val_rir
            )
        )
        # Create the custom 'merged' datasets
        self.dataset_train = AudioDatasetConvolvedWithRirDataset(
            self.dry_train,
            self.rir_dataset_train,
            pre_associate=not self.hparams.dynamic_mixing_at_training,
            dry_signal_target_len=self.hparams.dry_signal_target_len,
            rir_target_len=self.hparams.rir_target_len,
            align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
            dry_signal_start_index=self.hparams.dry_signal_start_index_train,
            convolve_here=not self.hparams.convolve_on_gpu
            and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
            resampling_transform=self.hparams.resampling_transform,
            normalize=self.hparams.normalize,
            ignore_silent_windows=self.hparams.ignore_silent_windows,
            rir_transforms_for_wet=self.hparams.rir_transforms_for_wet,
            dry_only_transforms=self.hparams.dry_only_transforms,
            rir_only_transforms=self.hparams.rir_only_transforms,
            index_according_to_rir_dataset=self.hparams.index_according_to_rir_dataset,
            enable_caching=self.hparams.enable_caching_train,
            limit_size=self.hparams.limit_training_size,
        )
        self.dataset_val = AudioDatasetConvolvedWithRirDataset(
            self.dry_val,
            self.rir_dataset_val,
            pre_associate=True,
            dry_signal_target_len=self.hparams.dry_signal_target_len,
            rir_target_len=self.hparams.rir_target_len,
            align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
            dry_signal_start_index=self.hparams.dry_signal_start_index_val_test,
            convolve_here=not self.hparams.convolve_on_gpu
            and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
            resampling_transform=self.hparams.resampling_transform,
            normalize=self.hparams.normalize,
            ignore_silent_windows=self.hparams.ignore_silent_windows,
            rir_transforms_for_wet=self.hparams.rir_transforms_for_wet,
            dry_only_transforms=self.hparams.dry_only_transforms,
            rir_only_transforms=self.hparams.rir_only_transforms,
            index_according_to_rir_dataset=self.hparams.index_according_to_rir_dataset,
            enable_caching=self.hparams.enable_caching_val,
            limit_size=1.0,
        )
        if self.rir_dataset_test is not None:
            self.dataset_test = AudioDatasetConvolvedWithRirDataset(
                self.dry_test,
                self.rir_dataset_test,
                pre_associate=True,
                dry_signal_target_len=self.hparams.dry_signal_target_len,
                rir_target_len=self.hparams.rir_target_len,
                align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
                dry_signal_start_index=self.hparams.dry_signal_start_index_val_test,
                convolve_here=not self.hparams.convolve_on_gpu
                and self.num_distinct_rirs_per_batch
                == self.hparams.batch_size,
                resampling_transform=self.hparams.resampling_transform,
                normalize=self.hparams.normalize,
                ignore_silent_windows=self.hparams.ignore_silent_windows,
                rir_transforms_for_wet=self.hparams.rir_transforms_for_wet,
                dry_only_transforms=self.hparams.dry_only_transforms,
                rir_only_transforms=self.hparams.rir_only_transforms,
                index_according_to_rir_dataset=self.hparams.index_according_to_rir_dataset,
                enable_caching=False,
                limit_size=1.0,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            prefetch_factor=(
                self.hparams.prefetch_factor
                if self.hparams.num_workers > 0
                else None
            ),
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=(
                self.hparams.prefetch_factor
                if self.hparams.num_workers > 0
                else None
            ),
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self):
        assert self.rir_dataset_test is not None
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=(
                self.hparams.prefetch_factor
                if self.hparams.num_workers > 0
                else None
            ),
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    @property
    def num_distinct_rirs_per_batch(self):
        if self.hparams.num_distinct_rirs_per_batch is None:
            return self.hparams.batch_size
        else:
            return self.hparams.num_distinct_rirs_per_batch

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # If num_distinct_rirs_per_batch is set
        if (
            isinstance(batch, (list, tuple))
            and self.num_distinct_rirs_per_batch < self.hparams.batch_size
        ):
            if self.hparams.batch_size % self.num_distinct_rirs_per_batch != 0:
                raise RuntimeError(
                    f"num_distinct_rirs_per_batch={self.num_distinct_rirs_per_batch} should divide batch size={batch.shape[0]}"
                )
            num_rirs_repeats = (
                self.hparams.batch_size // self.num_distinct_rirs_per_batch
            )
            batch[1][1] = batch[1][1][: self.num_distinct_rirs_per_batch, ...]
            if not self.hparams.convolve_on_gpu:
                batch[0] = self.convolution_transform(
                    batch[1][0],
                    batch[1][1].repeat_interleave(num_rirs_repeats, dim=0),
                )
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # to perform convolution on gpu
        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#on-after-batch-transfer
        if self.hparams.convolve_on_gpu and isinstance(batch, (list, tuple)):
            if self.num_distinct_rirs_per_batch < self.hparams.batch_size:
                # Check correctness
                num_rirs_repeats = (
                    self.hparams.batch_size // self.num_distinct_rirs_per_batch
                )
                batch[0] = self.convolution_transform(
                    batch[1][0],
                    batch[1][1].repeat_interleave(num_rirs_repeats, dim=0),
                )
            else:
                batch[0] = self.convolution_transform(batch[1][0], batch[1][1])
        # assert batch[0].isfinite().all()
        return batch


# %% Librispeech + synthethic


class LibriSpeechAudioOnlyDataset(LIBRISPEECH):
    """Same dataset as Librispeech, but returns audio only"""

    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


class LibrispeechSimulatedRirDataModule(
    AudioDatasetConvolvedWithRirDatasetDataModule
):
    """Librispeech convolved with Simulated RIRs"""

    def prepare_data(self):
        # Download librispeech selon config
        LibriSpeechAudioOnlyDataset(
            self.hparams.audio_root, url="train-clean-100", download=True
        )
        LibriSpeechAudioOnlyDataset(
            self.hparams.audio_root, url="test-clean", download=True
        )
        super().prepare_data()

    def split_audio_dataset(self, stage=None):
        # Pas besoin du stage, je n'ai rien de spécifique à certains stages (type transform ou augmentation)

        # split librispeech
        librispeech_train_full = LibriSpeechAudioOnlyDataset(
            self.hparams.audio_root, url="train-clean-100", download=False
        )
        proportions_librispeech = (
            1 - self.hparams.proportion_val_audio,
            self.hparams.proportion_val_audio,
        )
        self.dry_train, self.dry_val = random_split(
            librispeech_train_full, proportions_librispeech
        )
        self.dry_test = LibriSpeechAudioOnlyDataset(
            self.hparams.audio_root, url="test-clean", download=False
        )


class WSJDataset(torch.utils.data.Dataset):
    """WSJ0"""

    EXPECTED_SAMPLERATE = 16000
    TRAIN_TEST_DISKS = {
        "train": range(1, 13),
        "test": range(14, 16),
    }

    @property
    def wav_root(self):
        return os.path.join(
            self.audio_root,
            "WSJ",
            "WSJ0_wav_mic" + str(self.mic_number),
            self.subset,
        )

    @property
    def sphere_root(self):
        return os.path.join(self.audio_root, "WSJ", "csr_1")

    @property
    def base_path(self):
        if self.wav:
            return self.wav_root
        else:
            return self.sphere_root

    def _check_base_path_exists(self):
        if not os.path.isdir(self.base_path):
            raise ValueError("Path does not exist or is not WSJ0")

    def __init__(
        self,
        audio_root: str = "./data/speech",
        subset: str = "train",
        mic_number: int = 1,
        wav: bool = True,
    ):
        """
        Instantiate WSJ.

        Parameters
        ----------
        audio_root : str, optional
            DESCRIPTION. The default is "./data/speech".
        subset : str, optional
            "train" or "test". The default is "train".
        mic_number : int, optional
            DESCRIPTION. The default is 1.
        wav : bool, optional
            Whether to load the audio as a wav file (True) or the native sphere format (False).
            The default is True.

        """
        self.audio_root = audio_root
        self.subset = subset
        self.mic_number = mic_number
        self.wav = wav
        self.paths_list = []

        self._check_base_path_exists()

        if self.wav:
            self.paths_list = sorted(
                glob.glob(os.path.join(self.base_path, "*.wav"))
            )
        else:
            for i_disk in self.TRAIN_TEST_DISKS[subset]:
                self.paths_list.extend(
                    sorted(
                        glob.glob(
                            os.path.join(
                                self.base_path,
                                "11-" + str(i_disk) + ".1",
                                "**",
                                "*.wv" + str(self.mic_number),
                            ),
                            recursive=True,
                        )
                    )
                )

    @property
    def len_hours(self):
        total_len = 0
        for path in tqdm.tqdm(self.paths_list):
            total_len += torchaudio.info(path).num_frames
        return total_len / self.EXPECTED_SAMPLERATE / 3600

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        path = self.paths_list[index]
        x, fs = torchaudio.load(path)
        assert fs == self.EXPECTED_SAMPLERATE
        return x

    def export_to_wav(self, new_path=None):
        if new_path is None:
            new_path = self.wav_root
        os.makedirs(new_path)
        for i in tqdm.trange((len(self))):
            x = self[i]
            torchaudio.save(
                os.path.join(new_path, f"{i}.wav"), x, self.EXPECTED_SAMPLERATE
            )


WSJ0Dataset = WSJDataset

# %% WSJ + synthethic


class WSJ1Dataset(torch.utils.data.Dataset):
    """
    WSJ1 dataset.

    Test tasks are Hub 1 and 2, and spokes S1 to S4 and S9.
    S5 to S8 are excluded since they are noisy.

    """

    EXPECTED_SAMPLERATE = 16000
    TRAIN_TEST_DISKS = {
        "train": range(1, 32),
        "test": range(33, 35),
    }
    # we exclude s5-8 which deal with noisy data
    RELEVANT_TEST_TASKS = [
        "h1",
        "h2",
        "s1",
        "s2",
        "s3",
        "s4",
        "s9",
    ]

    @property
    def wav_root(self):
        return os.path.join(
            self.audio_root,
            "WSJ",
            "WSJ1_wav_mic" + str(self.mic_number),
            self.subset,
        )

    @property
    def sphere_root(self):
        return os.path.join(self.audio_root, "WSJ", "csr_2_comp")

    @property
    def base_path(self):
        if self.wav:
            return self.wav_root
        else:
            return self.sphere_root

    def _check_base_path_exists(self):
        if not os.path.isdir(self.base_path):
            raise ValueError("Path does not exist or is not WSJ1")

    def __init__(
        self,
        audio_root: str = "./data/speech",
        subset: str = "train",
        mic_number: int = 1,
        wav: bool = True,
    ):
        """
        Instantiate WSJ1 dataset.

        The test tasks are Hub 1 and 2, and spokes S1 to S4 and S9. S5 to S8 are excluded since they are noisy.

        Parameters
        ----------
        audio_root : str, optional
            Root in which WSJ folder can be found. The default is "./data/speech".
        subset : str, optional
            "train" or "test". The default is "train".
        mic_number : int, optional
            Mic number, 1 for headset (not reverberant). The default is 1.
        wav : bool, optional
            Whether data has been exported to wav beforehand. Else tries to load native sphere format. The default is True.

        """
        self.audio_root = audio_root
        self.subset = subset
        self.mic_number = mic_number
        self.wav = wav
        self.paths_list = []

        self._check_base_path_exists()

        if self.wav:
            self.paths_list = sorted(
                glob.glob(os.path.join(self.base_path, "*.wav"))
            )
        else:
            if "train" in subset.lower():
                with open(
                    os.path.join(
                        self.sphere_root,
                        "13-32.1/wsj1/doc/indices/wsj1/train/tr_s_wv1.ndx",
                    )
                ) as f:
                    self.paths_list.extend(
                        [
                            self.process_line(line)
                            for line in f
                            if not line.startswith(";;")
                        ]
                    )
                with open(
                    os.path.join(
                        self.sphere_root,
                        "13-32.1/wsj1/doc/indices/wsj1/train/tr_l_wv1.ndx",
                    )
                ) as f:
                    self.paths_list.extend(
                        [
                            self.process_line(line)
                            for line in f
                            if not line.startswith(";;")
                        ]
                    )
                # self.paths_list = list(dict.fromkeys(self.paths_list))  # remove duplicates
            else:
                # https://catalog.ldc.upenn.edu/docs/LDC94S13A/csrnov93.html
                relevant_index_files = os.listdir(
                    os.path.join(
                        self.sphere_root, "13-32.1/wsj1/doc/indices/wsj1/eval"
                    )
                )
                relevant_index_files = [
                    ndx
                    for ndx in relevant_index_files
                    if ndx.startswith(tuple(self.RELEVANT_TEST_TASKS))
                ]

                for ndx_file in relevant_index_files:
                    with open(
                        os.path.join(
                            self.sphere_root,
                            "13-32.1/wsj1/doc/indices/wsj1/eval",
                            ndx_file,
                        )
                    ) as f:
                        self.paths_list.extend(
                            [
                                self.process_line(line)
                                for line in f
                                if not line.startswith(";;")
                            ]
                        )
                # remove duplicates
                self.paths_list = list(dict.fromkeys(self.paths_list))

    def process_line(self, line):
        # wrong disk between 32 and 33 in test set
        if line.startswith("13_32_1:wsj1/si_et_") and not line.startswith(
            "13_32_1:wsj1/si_et_s9"
        ):
            line_list = list(line)
            line_list[4] = "3"
            line = "".join(line_list)
        if line.startswith("13_33_1:wsj1/si_et_s9"):
            line_list = list(line)
            line_list[4] = "2"
            line = "".join(line_list)
        line = line.rstrip()
        disk, end = line.split(":")
        d1, d2, d3 = disk.split("_")
        return self.sphere_root + "/" + d1 + "-" + d2 + ".1" + "/" + end

    @property
    def len_hours(self):
        total_len = 0
        for path in tqdm.tqdm(self.paths_list):
            total_len += torchaudio.info(path).num_frames
        return total_len / self.EXPECTED_SAMPLERATE / 3600

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        path = self.paths_list[index]
        x, fs = torchaudio.load(path)
        assert fs == self.EXPECTED_SAMPLERATE
        return x

    def export_to_wav(self, new_path=None):
        if new_path is None:
            new_path = self.wav_root
        os.makedirs(new_path)
        for i in tqdm.trange((len(self))):
            x = self[i]
            torchaudio.save(
                os.path.join(new_path, f"{i}.wav"), x, self.EXPECTED_SAMPLERATE
            )


class WSJSimulatedRirDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    NUM_OF_VALS = NotImplemented

    def split_audio_dataset(self, stage=None):
        self.wsj_train_full = WSJDataset(
            self.hparams.audio_root, subset="train"
        )
        self.dry_test = WSJDataset(self.hparams.audio_root, subset="test")
        if self.hparams.proportion_val_audio is not None:
            proportions_wsj = (
                1 - self.hparams.proportion_val_audio,
                self.hparams.proportion_val_audio,
            )
            self.dry_train, self.dry_val = random_split(
                self.wsj_train_full, proportions_wsj
            )
        else:
            self.dry_val = torch.utils.data.Subset(
                self.wsj_train_full, range(self.NUM_OF_VALS + 1)
            )
            self.dry_train = torch.utils.data.Subset(
                self.wsj_train_full,
                range(self.NUM_OF_VALS + 1, len(self.wsj_train_full)),
            )


class WSJ1SimulatedRirDataModule(
    AudioDatasetConvolvedWithRirDatasetDataModule
):
    NUM_OF_VALS = NotImplemented

    def split_audio_dataset(self, stage=None):
        self.wsj_train_full = WSJ1Dataset(
            self.hparams.audio_root, subset="train", wav=True
        )
        self.dry_test = WSJ1Dataset(
            self.hparams.audio_root, subset="test", wav=True
        )
        if self.hparams.proportion_val_audio is not None:
            proportions_wsj = (
                1 - self.hparams.proportion_val_audio,
                self.hparams.proportion_val_audio,
            )
            self.dry_train, self.dry_val = random_split(
                self.wsj_train_full, proportions_wsj
            )
        else:
            self.dry_val = torch.utils.data.Subset(
                self.wsj_train_full, range(self.NUM_OF_VALS + 1)
            )
            self.dry_train = torch.utils.data.Subset(
                self.wsj_train_full,
                range(self.NUM_OF_VALS + 1, len(self.wsj_train_full)),
            )


# %% Singing voice


class MedleyDBRawVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, fs=16000, base_path="data/music/MedleyDB_raw_voice"):
        self.base_path = base_path
        self.fs = fs
        self.resampling_transform = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=16000,
            resampling_method="sinc_interp_kaiser",
        )
        self.paths_list = sorted(
            glob.glob(os.path.join(self.base_path, "*.wav"))
        )

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        path = self.paths_list[index]
        x, fs = torchaudio.load(path)
        return self.resampling_transform(x)

    def extract_from_MedleyDB(
        self,
        original_path="/tsi/mir/MedleyDB/",
        singing_classes=["singer", "screamer", "rapper"],
    ):
        import shutil
        import yaml

        res_list = []
        song_dirs = os.listdir(original_path + "/Audio")
        for song_dir in song_dirs:
            print()
            print(song_dir)
            metadata_file = (
                original_path + "/Metadata/" + song_dir + "_METADATA.yaml"
            )
            with open(metadata_file) as f:
                metadata = yaml.load(f, yaml.loader.Loader)
            if "no" in metadata["has_bleed"]:
                raw_dir = metadata["raw_dir"]
                for v_stem in metadata["stems"].values():
                    for v_raw in v_stem["raw"].values():
                        for singing_class in singing_classes:
                            if singing_class in v_raw["instrument"]:
                                res_list.append(
                                    os.path.join(
                                        original_path,
                                        "Audio",
                                        song_dir,
                                        raw_dir,
                                        v_raw["filename"],
                                    )
                                )
            print({"has_bleed": metadata["has_bleed"]})
        # with open("raw_songs", "w") as f:
        #     print("\n".join(res_list), file=f)
        for source_file in res_list:
            print("copying", source_file)
            shutil.copy2(source_file, self.base_path)

    def split_by_song(self, *proportions):
        list_of_songs = list(
            set(
                [
                    os.path.basename(f).split("_RAW_")[0]
                    for f in self.paths_list
                ]
            )
        )
        separated_songs_subsets = random_split(
            list_of_songs, (1 - sum(proportions), *proportions)
        )
        split_datasets = []
        for song_subset in separated_songs_subsets:
            new_dataset = self.__class__(fs=self.fs, base_path=self.base_path)
            subset_song_names = [list_of_songs[i] for i in song_subset.indices]
            new_dataset.paths_list = [
                p
                for p in self.paths_list
                if any(song_name in p for song_name in subset_song_names)
            ]
            split_datasets.append(new_dataset)
        # we check that there is no song in comon
        return split_datasets


class MedleyDBRawVoiceRirDataModule(
    AudioDatasetConvolvedWithRirDatasetDataModule
):
    def split_audio_dataset(self, stage=None):
        self.audio_dataset = MedleyDBRawVoiceDataset(
            fs=16000, base_path="data/music/MedleyDB_raw_voice"
        )
        self.dataset_train, self.dataset_val, self.dataset_test = (
            self.audio_dataset.split_by_song(self.proportion_val_audio)
        )

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()


class MUSDB18(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_root: str = "./data/music",
        fs: int = 16000,
        subset: str = "train",
        return_stem=4,  # for voice
        return_channel=0,  # since we work with dereverb, we only want to return 1 channel. Averaging could remove the spatial audio processing that has been done on the raw voice
    ):
        self.audio_root = audio_root
        self.subset_dir = os.path.join(self.audio_root, "MUSDB18", subset)
        self.paths_list = sorted(
            glob.glob(os.path.join(self.subset_dir, "*.stem.mp4"))
        )
        self.resampling_transform = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=16000,
            resampling_method="sinc_interp_kaiser",
        )
        self.return_stem = return_stem
        self.return_channel = return_channel

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        x_np, fs = stempeg.read_stems(
            self.paths_list[index], stem_id=self.return_stem
        )
        y = torch.from_numpy(x_np[..., self.return_channel, None]).T
        return self.resampling_transform(y)


class MUSDB18ConvolvedWithRirDataModule(
    AudioDatasetConvolvedWithRirDatasetDataModule
):
    def __init__(self):
        raise NotImplementedError(
            "The data is assumed to be non-anechoic and hence could train the dereveb module directly"
        )


# %% ADASP


class AdaspRirDataset(torch.utils.data.Dataset):
    """See ADASP knowledge base."""

    ORIG_FS = 44100

    def __init__(
        self,
        datasets: list[str],
        fs: int = 16000,
        rt60_min=0.0,
        rt60_max=4.0,
    ):
        from adasp_data_management.reverb import ReverbCorpus

        self.reverb_corpus = ReverbCorpus()
        self.resampling_transform = torchaudio.transforms.Resample(
            orig_freq=self.ORIG_FS,
            new_freq=fs,
            resampling_method="sinc_interp_kaiser",
        )
        self.datasets = datasets
        # self.rir_properties = self.reverb_corpus.filter_database_with_ir_types(["rir"])
        self.rir_properties = self.reverb_corpus.filter_database_with_rt60(
            rt60_min=rt60_min, rt60_max=rt60_max
        )
        self.rir_properties = self.rir_properties.loc[
            self.rir_properties.ir_type == "rir"
        ]
        self.rir_properties.query("n_channels==1", inplace=True)
        self.rir_properties = self.rir_properties.loc[
            self.rir_properties.dataset.isin(self.datasets)
        ]

    def random_split_by_rooms(self, proportion_val_rir):
        if proportion_val_rir != 0.0 and proportion_val_rir != 1.0:
            raise NotImplementedError("Extract random rooms from each dataset")
        return random_split(self, [1 - proportion_val_rir, proportion_val_rir])

    def __len__(self):
        return len(self.rir_properties)

    def __getitem__(self, index):
        file_path = self.rir_properties.file_path.iloc[index]
        h, sample_orig_fs = torchaudio.load(file_path)
        assert sample_orig_fs == self.ORIG_FS
        h_resampled = self.resampling_transform(h)
        return_properties = dict()
        return h_resampled, return_properties


# %% Paired Data


class PairedDataset(torch.utils.data.Dataset):
    """
    Paired dataset

    Can be the result of `AudioDatasetConvolvedWithRirDataset.export`.
    Is meant to be used for testing.
    """

    def __init__(self, path="./data/test_wsj1_same"):
        self.path = path
        self.df = pd.read_csv(os.path.join(path, "properties.csv"))

    def __len__(self):
        return len(os.listdir(os.path.join(self.path, "wet")))

    def __getitem__(self, index):
        y, _ = torchaudio.load(
            os.path.join(self.path, "wet", str(index) + ".wav")
        )
        s, _ = torchaudio.load(
            os.path.join(self.path, "dry", str(index) + ".wav")
        )
        h, _ = torchaudio.load(
            os.path.join(self.path, "rir", str(index) + ".wav")
        )
        rir_properties = {
            k: torch.tensor(v).unsqueeze(0)
            for k, v in self.df.loc[index].to_dict().items()
            if k != "idx"
        }
        return y, (s, h, rir_properties)


class PairedDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    """
    DataModule associated with paired dataset

    Can be the result of `AudioDatasetConvolvedWithRirDataset.export`.
    Only defines test dataset.
    """

    def __init__(self, path="./data/paired_test_same", *args, **kwargs):
        super(L.LightningDataModule, self).__init__()
        self.path = path

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset_test = PairedDataset(self.path)

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch


# %% EARS


class EARSReverbDataset(torch.utils.data.Dataset):
    """`EARS-Reverb <https://doi.org/10.21437/Interspeech.2024-153>`_ dataset."""

    ORIG_FS = 48000

    def __init__(
        self,
        root_path: str = "./data/speech/EARS-Reverb/train",
        fs: int = 16000,
        target_len: float = 4.0,
        enable_caching: bool = False,
        deterministic_crop: bool = False,
        zero_pad: bool = True,
        return_rir: bool = False,
        resample_rir: bool = True,
    ):
        """
        Instantiate EARSReverbDataset.

        Parameters
        ----------
        root_path : str, optional
            Path of EARS-Reverb. The default is "./data/speech/EARS-Reverb/train".
        fs : int, optional
            Sampling rate of dry and wet audios.
            Uses `torchaudio.transforms.Resample` with `sinc_interp_kaiser`
            to resample from 48 kHz to fs.
            The default is 16000.
        target_len : float, optional
            Target length of the audios in seconds.
            The default is 4.0 and matches the value used at training.
        enable_caching : bool, optional
            Whether to cache audios in CPU RAM. The default is False.
        deterministic_crop : bool, optional
            Whether to start the audio at the beginning or at a random index.
            The default is False.
        zero_pad : bool, optional
            Whether to zero-pad or crop the audio to taget_len. The default is True.
        return_rir : bool, optional
            Whether to return the RIR used to synthesize wet from dry.
            Note that wet is not recomputed.
            The default is False.
        resample_rir : bool, optional
            Whether to resample the RIR (if it is returned) to match dry and wet audio fs.
            If False, the RIR is returned without resampling at 48 KHz (as in the original dataset).
            The default is True.

        """
        self.fs = fs
        self.root_path = root_path
        self.enable_caching = enable_caching
        self.target_len = target_len
        self.deterministic_crop = deterministic_crop
        self.zero_pad = zero_pad
        self.return_rir = return_rir
        self.resample_rir = resample_rir

        if self.return_rir:
            self.dry_rir_df = pd.read_csv(
                f"{self.root_path}_unique_rirs/pairs.csv",
                index_col="speech_idx",
            )

        self.target_len_samples = int(self.target_len * self.fs)
        self.rir_target_len_samples = (
            self.target_len_samples
            if self.resample_rir
            else round(self.ORIG_FS / self.fs * self.target_len_samples)
        )

        self.resampling_transform = torchaudio.transforms.Resample(
            orig_freq=self.ORIG_FS,
            new_freq=self.fs,
            resampling_method="sinc_interp_kaiser",
        )

        self.wet_paths = sorted(
            glob.glob(
                os.path.join(root_path, "reverberant/**/*.wav"),
                recursive=True,
            )
        )
        self.dry_paths, self.rt_60s = zip(
            *[
                self._get_dry_path_and_rt60_from_wet_path(wp)
                for wp in self.wet_paths
            ]
        )

        if self.enable_caching:
            if self.deterministic_crop and self.zero_pad:
                self.dry_cache = torch.full(
                    (len(self.wet_paths), 1, self.target_len_samples),
                    torch.nan,
                )
                self.wet_cache = torch.full(
                    (len(self.wet_paths), 1, self.target_len_samples),
                    torch.nan,
                )
            else:
                self.dry_cache = dict()
                self.wet_cache = dict()
            if self.return_rir:
                self.rir_cache = torch.full(
                    (
                        self.dry_rir_df.rir_idx.nunique(),
                        1,
                        self.rir_target_len_samples,
                    ),
                    torch.nan,
                )
            # We need to cache before the first epoch, before the creation of the dataloaders
            # or else it should have num_workers=0 during first epoch which is bad
            # https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/5
            print("Caching dataset")
            self._cache_all_audios()

    def __len__(self):
        return len(self.wet_paths)

    def _cache_all_audios(self):
        for i in tqdm.trange(len(self)):
            _ = self[i]

    @classmethod
    def _get_dry_path_and_rt60_from_wet_path(self, wet_path):
        rt_60 = float(wet_path[-8:-4])
        dry_path = wet_path[:-9].replace("reverberant", "clean") + ".wav"
        return dry_path, rt_60

    def _load_rir(self, index):
        if self.return_rir:
            speech_idx = int(self.dry_paths[index][-9:-4])
            # get the rir path
            rir_idx, rir_expected_rt_60 = self.dry_rir_df.loc[speech_idx]
            rir_idx = int(rir_idx)
            assert (
                abs(rir_expected_rt_60 - self.rt_60s[index])
                < 0.02  # to alleviate dataframe rounding errors
            ), f"rir ({rir_expected_rt_60}) and wet ({self.rt_60s[index]}) rt_60s don't match"
            # load the rir
            if self.enable_caching:
                try:
                    h = self.rir_cache[rir_idx]
                    if h.isnan().any():
                        raise KeyError()
                except:
                    h = self._load_and_resample_rir_from_file(rir_idx)
                    self.rir_cache[rir_idx] = h
            else:
                h = self._load_and_resample_rir_from_file(rir_idx)
        else:
            h = torch.full((1, 0), torch.nan)
        return h

    def _load_and_resample_rir_from_file(self, rir_idx):
        rir_at_48kHz, fs_should_be_48khz = torchaudio.load(
            f"{self.root_path}_unique_rirs/rir_{rir_idx}.wav"
        )
        assert fs_should_be_48khz == self.ORIG_FS
        if self.resample_rir:
            h = self.resampling_transform(rir_at_48kHz)
        else:
            h = rir_at_48kHz
        h = zero_pad(h, self.rir_target_len_samples)
        return h

    def _load_wet_dry(self, index):
        if self.enable_caching:
            try:
                y, s = self.wet_cache[index], self.dry_cache[index]
                if y.isnan().any():
                    raise KeyError()
                y_cropped, s_cropped = self._crop_signals_if_necessary(y, s)
            except KeyError:
                y, s = self._load_and_resample_wet_dry_from_file(index)
                if self.deterministic_crop:
                    # we crop then put in cache
                    y_cropped, s_cropped = self._crop_signals_if_necessary(
                        y, s
                    )
                    self.dry_cache[index] = s_cropped
                    self.wet_cache[index] = y_cropped
                else:
                    # we put in cache uncropped then we crop
                    self.dry_cache[index] = s
                    self.wet_cache[index] = y
                    y_cropped, s_cropped = self._crop_signals_if_necessary(
                        y, s
                    )
        else:
            y, s = self._load_and_resample_wet_dry_from_file(index)
            y_cropped, s_cropped = self._crop_signals_if_necessary(y, s)
        return y_cropped, s_cropped

    def _load_and_resample_wet_dry_from_file(self, index):
        dry_signal_at_48k, fs_should_be_48khz = torchaudio.load(
            self.dry_paths[index]
        )
        wet_signal_at_48k, fs_should_be_48khz = torchaudio.load(
            self.wet_paths[index]
        )
        assert fs_should_be_48khz == self.ORIG_FS
        dry_signal_at_fs = self.resampling_transform(dry_signal_at_48k)
        wet_signal_at_fs = self.resampling_transform(wet_signal_at_48k)
        return (wet_signal_at_fs, dry_signal_at_fs)

    def _crop_signals_if_necessary(self, wet, dry):
        assert wet.size(-1) == dry.size(-1)
        if self.deterministic_crop:
            begin_idx = 0
        else:
            begin_idx = torch.randint(
                wet.size(-1) - self.target_len_samples, size=(1,)
            )[0]
        wet_cropped = wet[..., begin_idx : begin_idx + self.target_len_samples]
        dry_cropped = dry[..., begin_idx : begin_idx + self.target_len_samples]
        if self.zero_pad:
            wet_cropped = zero_pad(wet_cropped, self.target_len_samples)
            dry_cropped = zero_pad(dry_cropped, self.target_len_samples)
        return wet_cropped, dry_cropped

    def __getitem__(self, index):
        y_cropped, s_cropped = self._load_wet_dry(index)
        h = self._load_rir(index)
        rt_60 = {"rt_60": torch.tensor(self.rt_60s[index])}
        return y_cropped, (s_cropped, h, rt_60)

    @property
    def len_hours(self):
        """Computes the cumulated duration of audios from the dataset in hours"""
        total_len_hours = 0
        for path in tqdm.tqdm(
            self.dry_paths, postfix=f"{total_len_hours:.2f}"
        ):
            total_len_hours += (
                torchaudio.info(path).num_frames / self.ORIG_FS / 3600
            )
        return total_len_hours

    def export_unique_rirs(
        self,
        rir_folder="/home/ids/lbahrman/ears_benchmark/raw_datadir",
        target_sr=48000,
    ):
        """
        Extract the RIRs used at dataset creation.

        From `this script <https://github.com/sp-uhh/ears_benchmark/blob/main/generate_ears_reverb.py>`_.
        """
        df = pd.read_csv(self.root_path + ".csv", index_col="id")

        dry_rir_mapping = dict()
        os.makedirs(self.root_path + "_unique_rirs", exist_ok=True)

        import sofa
        import mat73
        from soundfile import read, write
        from librosa import resample
        from scipy import stats

        def calc_rt60(h, sr=48000, rt="t30"):
            """
            RT60 measurement routine acording to Schroeder's method [1].

            [1] M. R. Schroeder, "New Method of Measuring Reverberation Time," J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.

            Adapted from https://github.com/python-acoustics/python-acoustics/blob/99d79206159b822ea2f4e9d27c8b2fbfeb704d38/acoustics/room.py#L156
            """
            rt = rt.lower()
            if rt == "t30":
                init = -5.0
                end = -35.0
                factor = 2.0
            elif rt == "t20":
                init = -5.0
                end = -25.0
                factor = 3.0
            elif rt == "t10":
                init = -5.0
                end = -15.0
                factor = 6.0
            elif rt == "edt":
                init = 0.0
                end = -10.0
                factor = 6.0

            h_abs = np.abs(h) / np.max(np.abs(h))

            # Schroeder integration
            sch = np.cumsum(h_abs[::-1] ** 2)[::-1]
            sch_db = 10.0 * np.log10(sch / np.max(sch) + 1e-20)

            # Linear regression
            sch_init = sch_db[np.abs(sch_db - init).argmin()]
            sch_end = sch_db[np.abs(sch_db - end).argmin()]
            init_sample = np.where(sch_db == sch_init)[0][0]
            end_sample = np.where(sch_db == sch_end)[0][0]
            x = np.arange(init_sample, end_sample + 1) / sr
            y = sch_db[init_sample : end_sample + 1]
            slope, intercept = stats.linregress(x, y)[0:2]

            # Reverberation time (T30, T20, T10 or EDT)
            db_regress_init = (init - intercept) / slope
            db_regress_end = (end - intercept) / slope
            t60 = factor * (db_regress_end - db_regress_init)
            return t60

        rir_idx = 0
        seen_rir_files_and_channel = dict()
        for dry_idx in tqdm.tqdm(df.index):
            df_current_idx = dry_idx
            rir_file = rir_folder + df.loc[df_current_idx].rir_file
            channel = df.loc[df_current_idx].channel
            if (rir_file, channel) not in seen_rir_files_and_channel.keys():
                if "ARNI" in rir_file:
                    rir, sr = read(rir_file, always_2d=True)
                    rir = rir[:, channel]
                    assert sr == 44100, f"Sampling rate of {rir_file} is {sr}"
                    rir = resample(rir, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                elif rir_file.endswith(".wav"):
                    rir, sr = read(rir_file, always_2d=True)
                    rir = rir[:, channel]
                elif rir_file.endswith(".sofa"):
                    hrtf = sofa.Database.open(rir_file)
                    rir = hrtf.Data.IR.get_values()
                    rir = rir[0, channel, :]
                    sr = hrtf.Data.SamplingRate.get_values().item()
                elif rir_file.endswith(".mat"):
                    rir = mat73.loadmat(rir_file)
                    sr = rir["fs"].item()
                    rir = rir["data"]
                    rir = rir[:, channel]
                else:
                    raise ValueError(f"Unknown file format: {rir_file}")

                assert sr == target_sr, f"Sampling rate of {rir_file} is {sr}"

                # Cut RIR to get direct path at the beginning
                max_index = np.argmax(np.abs(rir))
                rir = rir[max_index:]

                # Normalize RIRs in range [0.1, 0.7]
                if np.max(np.abs(rir)) < 0.1:
                    rir = 0.1 * rir / np.max(np.abs(rir))
                elif np.max(np.abs(rir)) > 0.7:
                    rir = 0.7 * rir / np.max(np.abs(rir))

                rt60 = calc_rt60(rir, sr=sr)

                write(
                    f"{self.root_path}_unique_rirs/rir_{rir_idx}.wav",
                    rir,
                    target_sr,
                )

                seen_rir_files_and_channel[(rir_file, channel)] = (
                    rir_idx,
                    rt60,
                )
                rir_idx += 1

            dry_rir_mapping[dry_idx] = seen_rir_files_and_channel[
                (rir_file, channel)
            ]
        pd.DataFrame.from_dict(
            dry_rir_mapping, orient="index", columns=["rir_idx", "rt_60"]
        ).to_csv(
            f"{self.root_path}_unique_rirs/pairs.csv",
            float_format="%.3f",
            index_label="speech_idx",
        )


class EarsOnlyRIRsDataset(torch.utils.data.Dataset):
    """
    Only the RIRs used in EARS-Reverb.

    Now included in `EARSReverbDataset`

    RIRs should have been exported beforehand using `EARSReverbDataset.export_unique_rirs`
    """

    ORIG_FS = 48000

    def __init__(self, rir_root, fs=16000):
        self.fs = fs
        self.rir_root = rir_root
        self.rir_list = sorted(
            glob.glob(os.path.join(rir_root, "*.wav"), recursive=True)
        )
        if self.fs != self.ORIG_FS:
            self.resampling_transform = torchaudio.transforms.Resample(
                orig_freq=self.ORIG_FS,
                new_freq=self.fs,
                resampling_method="sinc_interp_kaiser",
            )
        else:
            self.resampling_transform = torch.nn.Identity()

    def __len__(self):
        return len(self.rir_list)

    def __getitem__(self, index):
        rir_at_48k, fs_should_be_48khz = torchaudio.load(self.rir_list[index])
        assert fs_should_be_48khz == self.ORIG_FS
        rir_resampled = self.resampling_transform(rir_at_48k)
        return rir_resampled, {}


class EARSReverbDrySpeechOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset containing only dry speech from EARS-Reverb.

    Meant to be used with SynthethicRIRs.
    """

    ORIG_FS = 48000

    def __init__(self, audio_root="./data/speech/", subset="train", fs=16000):
        self.resampling_transform = (
            torchaudio.transforms.Resample(
                orig_freq=self.ORIG_FS,
                new_freq=fs,
                resampling_method="sinc_interp_kaiser",
            )
            if self.ORIG_FS != fs
            else torch.nn.Identity()
        )
        self.dry_paths = sorted(
            glob.glob(
                os.path.join(
                    audio_root, "EARS-Reverb", subset, "clean/**/*.wav"
                ),
                recursive=True,
            )
        )

    def __len__(self):
        return len(self.dry_paths)

    def __getitem__(self, index):
        s, fs = torchaudio.load(self.dry_paths[index])
        if fs != self.ORIG_FS:
            raise RuntimeError("Wrong fs")
        s_resampled = self.resampling_transform(s)
        return s_resampled


class EARSSimulatedRirsDataModule(
    AudioDatasetConvolvedWithRirDatasetDataModule
):
    """EARS-Reverb dry speech only + Simulated RIRs"""

    def split_audio_dataset(self, stage=None):
        self.dry_train = EARSReverbDrySpeechOnlyDataset(
            self.hparams.audio_root, subset="train"
        )
        self.dry_val = EARSReverbDrySpeechOnlyDataset(
            self.hparams.audio_root, subset="valid"
        )
        self.dry_test = EARSReverbDrySpeechOnlyDataset(
            self.hparams.audio_root, subset="test"
        )


class EARSReverbDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    """Datamodule for EARSReverbDataset"""

    def __init__(
        self,
        root_path: str = "./data/speech/EARS-Reverb",
        fs: int = 16000,
        target_len_train_val: float = 4.0,
        enable_caching_train: bool = False,
        enable_caching_val: bool = True,
        batch_size: int = 4,
        return_rir: bool = False,
        resample_rir: bool = True,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        limit_training_size: float | int = 1.0,
    ):
        super(AudioDatasetConvolvedWithRirDatasetDataModule, self).__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # In this case, since the dataset is already defined without any need to convolve it with RIRs, we redefine setupe
        if stage is None or stage == "fit":
            self.dataset_train = EARSReverbDataset(
                root_path=os.path.join(self.hparams.root_path, "train"),
                fs=self.hparams.fs,
                target_len=self.hparams.target_len_train_val,
                enable_caching=self.hparams.enable_caching_train,
                deterministic_crop=False,
                zero_pad=True,
                return_rir=self.hparams.return_rir,
                resample_rir=self.hparams.resample_rir,
            )
            self.dataset_train = limit_dataset_size(
                self.dataset_train, self.hparams.limit_training_size
            )
            self.dataset_val = EARSReverbDataset(
                root_path=os.path.join(self.hparams.root_path, "valid"),
                fs=self.hparams.fs,
                target_len=self.hparams.target_len_train_val,
                enable_caching=self.hparams.enable_caching_val,
                deterministic_crop=True,
                zero_pad=True,
                return_rir=self.hparams.return_rir,
                resample_rir=self.hparams.resample_rir,
            )
        # Never need cache, 30sec target len as in the paper presenting EARS-benchmark
        self.dataset_test = EARSReverbDataset(
            root_path=os.path.join(self.hparams.root_path, "test"),
            fs=self.hparams.fs,
            target_len=30.0,
            enable_caching=False,
            deterministic_crop=True,
            zero_pad=False,
            return_rir=self.hparams.return_rir,
            resample_rir=self.hparams.resample_rir,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            prefetch_factor=(
                self.hparams.prefetch_factor
                if self.hparams.num_workers > 0
                else None
            ),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            prefetch_factor=(
                self.hparams.prefetch_factor
                if self.hparams.num_workers > 0
                else None
            ),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            prefetch_factor=(
                self.hparams.prefetch_factor
                if self.hparams.num_workers > 0
                else None
            ),
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch


# wsj1_reverb_test=AudioDatasetConvolvedWithRirDataset(audio_dataset=WSJ1Dataset("data/speech", subset="test"), rir_dataset)
def generate_paired_wsj_reverb_dataset(output_dataset="data/wsj_reverb"):
    wsj = WSJ1Dataset("data/speech", subset="test", wav=True)
    df_ears_rirs = pd.read_csv(
        "data/speech/EARS-Reverb/test_unique_rirs/pairs.csv",
        index_col="speech_idx",
    )
    wsj_transcripts = pd.read_csv(
        "data/speech/WSJ/WSJ1_wav_mic1/test_transcripts.csv", index_col="wav"
    )
    rirs_path = os.path.join(
        "data", "speech", "EARS-Reverb", "test_unique_rirs"
    )

    # first check that same rir_idx always have the same rt60
    assert df_ears_rirs.groupby("rir_idx")["rt_60"].nunique().eq(1).all()
    # we now extract only the series "rir_idx -> rt_60"
    unique_rir_properties = df_ears_rirs.groupby("rir_idx")["rt_60"].first()

    # properties should contain both rt60 and transcript
    rng = np.random.default_rng(42)
    dry_idx_to_rir_array = rng.choice(
        unique_rir_properties.index, size=len(wsj_transcripts)
    )
    dry_path_to_rir_series = pd.Series(
        dry_idx_to_rir_array, index=wsj_transcripts.index, name="rir_idx"
    )
    wsj_transcripts["rir_idx"] = dry_path_to_rir_series
    full_properties = wsj_transcripts.join(
        unique_rir_properties, on="rir_idx", how="left"
    )
    # check that we didn't change any rt_60
    assert full_properties.groupby("rir_idx")["rt_60"].nunique().eq(1).all()

    os.makedirs(os.path.join(output_dataset, "dry"))
    os.makedirs(os.path.join(output_dataset, "wet"))
    os.makedirs(os.path.join(output_dataset, "rir"))
    full_properties.to_csv(os.path.join(output_dataset, "properties.csv"))

    resampling_transform = torchaudio.transforms.Resample(
        48000, 16000, resampling_method="sinc_interp_kaiser"
    )
    convolve = torchaudio.transforms.FFTConvolve()
    for idx, (dry_path, properties) in enumerate(
        tqdm.tqdm(full_properties.iterrows())
    ):
        s, fs = torchaudio.load(dry_path)
        rir_idx = properties["rir_idx"]
        h, fs_orig = torchaudio.load(
            os.path.join(rirs_path, f"rir_{rir_idx}.wav")
        )
        assert fs_orig == resampling_transform.orig_freq
        h_resampled = resampling_transform(h)
        y = convolve(s, h)
        torchaudio.save(os.path.join(output_dataset, "dry", f"{idx}.wav"), s)
        torchaudio.save(os.path.join(output_dataset, "wet", f"{idx}.wav"), y)
        torchaudio.save(os.path.join(output_dataset, "rir", f"{idx}.wav"), y)


# %% Additional functions


class NoDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    def __init__(self, **kwargs):
        pass


def reset_batch_size(dataloader, new_batch_size):
    """
    Resets the batch size of a dataloader.

    Since this attribute cannot be modified inplace, a new dataloader is returned.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader.
    new_batch_size : int
        New batch size.

    Returns
    -------
    torch.utils.data.DataLoader
        new dataloader with new batch size.

    """
    return torch.utils.data.DataLoader(
        dataset=dataloader.dataset,
        batch_size=new_batch_size,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device,
    )


def merge_synthethic_datasets(
    dataset_1_root: str = "./data/rirs_dar_train_singlerirs",
    dataset_2_root: str = "./data/rirs_dar_train",
    save_path: str = "./data/rirs_dar_merged",
    limit: str = 203200,
):
    """
    Merge 2 synthetic RIRs datasets.

    Parameters
    ----------
    dataset_1_root : str, optional
        Dataset 1 root. The default is "./data/rirs_dar_train_singlerirs".
    dataset_2_root : str, optional
        Dataset 2 root. The default is "./data/rirs_dar_train".
    save_path : str, optional
        Where to store the new concatenated csv file. The default is "./data/rirs_dar_merged".
    limit : str, optional
        Max size of the returned dataset. The default is 203200.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame used internally in SynthethicRirDataset.

    """
    properties_1 = pd.read_csv(
        os.path.join(dataset_1_root, "properties.csv"),
        index_col="rir_global_idx",
    )
    properties_2 = pd.read_csv(
        os.path.join(dataset_2_root, "properties.csv"),
        index_col="rir_global_idx",
    )
    max_room_1 = properties_1.room_idx.max()
    properties_2.room_idx += max_room_1 + 1
    df = pd.concat([properties_1, properties_2], ignore_index=True)
    df.index.name = properties_1.index.name
    if limit:
        df = df.iloc[:limit]
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "properties.csv"), float_format="%.3f")
    return df


def make_test_dataset(dataset_name="test_wsj1_same"):
    # Supported: paired_medleydb_same, paired_wsj1_same
    from lightning.pytorch import seed_everything
    from model.reverb_models.early_echoes import FixedTimeEarlyEnd

    seed_everything(12)

    dataset_path = os.path.join("./data/", dataset_name)
    if os.path.isdir(dataset_path):
        os.rename(dataset_path, dataset_path + "-old")

    if "same" in dataset_name:
        rir_dataset_test = SynthethicRirDataset(
            # rir_root="./data/rirs_dar_merged",
            rir_root="./data/rirs_v2_test_same",
            # num_new_rooms=500,
        )
    elif "air" in dataset_name.lower():
        rir_dataset_test = AdaspRirDataset(datasets=["AIR", "openAIR"])
    else:
        raise NotImplementedError(f"{dataset_name} not supported")
    if "early" in dataset_name.lower():
        dry_only_transforms = ConvolveDryWithEarly(
            early_echoes_masking_module=FixedTimeEarlyEnd(end_time=0.020)
        )
    else:
        dry_only_transforms = None

    if "wsj1" in dataset_name:
        data_module = WSJ1SimulatedRirDataModule(
            batch_size=1,
            dry_signal_target_len=49151,
            rir_dataset=rir_dataset_test,
            rir_dataset_test=rir_dataset_test,
            dry_signal_start_index_train=None,
            dry_signal_start_index_val_test=16000,
            proportion_val_audio=0.0,
            proportion_val_rir=0.0,
            index_according_to_rir_dataset=False,
            align_and_scale_to_direct_path=True,
            dry_only_transforms=dry_only_transforms,
        )
    elif "medleydb" in dataset_name.lower():
        data_module = MedleyDBRawVoiceRirDataModule(
            batch_size=1,
            dry_signal_target_len=49151,
            rir_dataset=rir_dataset_test,
            rir_dataset_test=rir_dataset_test,
            dry_signal_start_index_train=None,
            dry_signal_start_index_val_test=16000,
            proportion_val_audio=0.0,
            proportion_val_rir=0.0,
            index_according_to_rir_dataset=False,
            align_and_scale_to_direct_path=True,
            dry_only_transforms=dry_only_transforms,
        )
    elif "ears" in dataset_name.lower() and "same" in dataset_name.lower():
        data_module = EARSSimulatedRirsDataModule(
            batch_size=1,
            dry_signal_target_len=None,
            rir_dataset=rir_dataset_test,
            rir_dataset_test=rir_dataset_test,
            dry_signal_start_index_train=0,
            dry_signal_start_index_val_test=0,
            proportion_val_audio=0.0,
            proportion_val_rir=0.0,
            index_according_to_rir_dataset=False,
            align_and_scale_to_direct_path=True,
        )
    else:
        raise NotImplementedError(f"{dataset_name} not supported")

    data_module.prepare_data()
    data_module.setup()

    data_module.dataset_test.export(dataset_path, crop=None)

    import tarfile

    print("compressing")
    with tarfile.open(dataset_path + ".tgz", "w:gz") as tar:
        tar.add(dataset_path, arcname=os.path.basename(dataset_path))


def compare_drr_augmentations():
    import matplotlib.pyplot as plt
    from model.reverb_models.polack import RirToPolack
    from model.reverb_models.early_echoes import MeanFreePathEarlyEnd

    # rir_dataset = SynthethicRirDataset(
    #     rir_root="./data/rirs_v2_val_hard",
    #     # num_new_rooms=500,
    #     num_new_rooms=10,
    #     room_dim_range=(10.0, 15.0),
    #     room_height_range=(4.0, 6.0),
    #     rt60_range=(1.0, 1.5),
    #     num_sources_per_room=1,
    #     num_mics_per_room=16,
    #     min_distance_to_wall=0.5,
    #     mic_height_range=(0.7, 3.5),  # also used for source placement
    #     source_mic_distance_range=(2.5, 4.0),
    # )
    # rir_dataset.generate_data_if_needed()
    # merge_synthethic_datasets()

    WSJ1 = WSJ1Dataset("data/speech/", subset="train", wav=True)
    # print(WSJ1.len_hours())
    # WSJ1.export_to_wav()
    # """
    rir_dataset = SynthethicRirDataset(
        rir_root="./data/rirs_dar_merged",
        # num_new_rooms=500,
    )
    rir_dataset.generate_data_if_needed()

    from lightning.pytorch import seed_everything

    seed_everything(12)

    # rir_dataset = AdaspRirDataset(datasets=["AIR", "openAIR"])
    data_module = WSJSimulatedRirDataModule(
        batch_size=8,
        rir_dataset=rir_dataset,
        dry_signal_start_index_train=None,
        proportion_val_audio=None,
        proportion_val_rir=0.01,
        index_according_to_rir_dataset=True,
        # rir_further_transforms=DARRirAugmentation(),
        # rir_further_transforms=RirToPolack(early_echoes_masking_module=MeanFreePathEarlyEnd()),
        # dry_only_transforms=ConvolveDryWithEarly(early_echoes_masking_module=MeanFreePathEarlyEnd()),
    )
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.val_dataloader()
    y, (x, rir, rir_properties) = next(iter(loader))

    plt.figure()
    plt.plot(rir[0].squeeze(), label="RIR")
    plt.plot(
        RandomModifyDRR()(rir.clone())[0].squeeze(),
        label="with DRR augmentation",
    )
    plt.plot(
        DARRirAugmentation()(rir.clone())[0].squeeze(),
        label="Full DAR augmentation",
    )
    plt.plot(
        RIRToLate(early_echoes_masking_module=MeanFreePathEarlyEnd())(
            rir.clone(), rir_properties
        )[0].squeeze(),
        label="late reverb",
    )
    plt.legend()


def test_ears_reverb():
    for enable_caching_train, return_rir in itertools.product(
        (
            # False,
            True,
        ),
        (
            # False,
            True,
        ),
    ):
        print(
            f"enable_caching_train={enable_caching_train}, return_rir={return_rir}"
        )
        data_module = EARSReverbDataModule(
            enable_caching_train=enable_caching_train,
            return_rir=return_rir,
            num_workers=8,
        )
        data_module.prepare_data()
        data_module.setup()
        loader = data_module.train_dataloader()
        print("train")
        y, (x, rir, rir_properties) = next(iter(loader))
        print(rir.shape, rir_properties["rt_60"].shape)

        print("val")
        loader = data_module.val_dataloader()
        y, (x, rir, rir_properties) = next(iter(loader))
        print(rir.shape, rir_properties["rt_60"].shape)

    print("testing caching")
    data_module.dataset_train._cache_all_audios()
    print("first loader pass")
    for _ in tqdm.tqdm(
        data_module.train_dataloader(),
        total=len(data_module.dataset_train) // data_module.hparams.batch_size,
    ):
        pass
    print("second loader pass")
    for _ in tqdm.tqdm(
        data_module.train_dataloader(),
        total=len(data_module.dataset_train) // data_module.hparams.batch_size,
    ):
        pass


def test_ears_simulated():
    rir_dataset = SynthethicRirDataset(
        rir_root="./data/rirs_v2", return_properties=[]
    )
    data_module = EARSSimulatedRirsDataModule(
        rir_dataset=rir_dataset,
        audio_root="./data/speech",
        dry_signal_target_len=64000,
        rir_target_len=32000,
        align_and_scale_to_direct_path=True,
        dry_signal_start_index_train=None,
        dry_signal_start_index_val_test=0,
        proportion_val_audio=None,
        proportion_val_rir=0.1,
        num_workers=8,
        convolve_on_gpu=True,
        normalize=True,
        ignore_silent_windows=True,
        enable_caching_train=True,
        enable_caching_val=True,
    )
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()
    print("train")
    y, (x, rir, rir_properties) = next(iter(loader))
    print("first loader pass")
    for _ in tqdm.tqdm(
        data_module.train_dataloader(),
        total=len(data_module.dataset_train) // data_module.hparams.batch_size,
    ):
        pass
    print("second loader pass")
    for _ in tqdm.tqdm(
        data_module.train_dataloader(),
        total=len(data_module.dataset_train) // data_module.hparams.batch_size,
    ):
        pass


# %% Main
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test_ears_reverb()
    # test_ears_simulated()

    # compare_drr_augmentations()
    # make_test_dataset(dataset_name="test_ears_same")
    # make_test_dataset(dataset_name="test_wsj1_same_earlytgt")
    # make_test_dataset(dataset_name="test_wsj1_air_earlytgt")
    # make_test_dataset(dataset_name="test_medleydb_same_earlytgt")
