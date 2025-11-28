#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:45:37 2024

@author: louis
"""
import torch
import math
from torch import nn
import torchaudio
from model.utils.tensor_ops import energy_to_db, arange_last_dim_like, Polynomial
import itertools
import scipy
import matplotlib.pyplot as plt


class RT60Estimator(nn.Module): ...


class RT60Prego(RT60Estimator):
    def __init__(
        self,
        polynomial_coeffs: list | tuple | None = None,  # coeffs if already set
        polynomial_deg: int | None = 1,  # degree if deg not set
        minimal_L_lim=3,
        fs=16000,
        default_rt_60: float = 0.5,
        edc_regression_intersect_zero: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        self.fs = fs
        self.M = int(0.05 * self.fs)  # Win length
        self.K = 2 ** math.ceil(math.log2(self.M))  # n_fft
        self.V = self.M // 4  # Overlapping samples
        self.num_cropped_bins = (4000 * self.K) // fs
        self.orig_L_lim = round(0.500 * self.fs / (self.M - self.V))
        self.minimal_L_lim = minimal_L_lim
        self.minimal_db_decrease = 10.0
        self.default_rt_60 = default_rt_60
        self.edc_regression_intersect_zero = edc_regression_intersect_zero

        self.stft_module = torchaudio.transforms.Spectrogram(
            n_fft=self.K,
            win_length=self.M,
            hop_length=self.M - self.V,
            power=2.0,
        )
        self.stft_step_rate = self.M - self.V
        self.polynomial = Polynomial(
            coeffs=polynomial_coeffs,
            deg=polynomial_deg,
            trainable=trainable,
        )

    @property
    def trainable(self):
        return self.polynomial.trainable

    def compute_batched_mask(self, Y):
        full_mask = Y.bool().zero_()
        for L_lim in range(self.orig_L_lim, self.minimal_L_lim - 1, -1):
            # Unfold to get candidate sliding windows
            Y_unfolded = Y.unfold(-1, L_lim, 1)
            # Check if the energy is decreasing in the sliding windowed STFT
            fdr_mask_start = (Y_unfolded.diff(dim=-1) < 0).all(dim=-1, keepdim=True)

            # Complete the mask
            fdr_mask_unfolded = fdr_mask_start.expand(-1, -1, -1, -1, L_lim).reshape(
                -1, fdr_mask_start.shape[-2], L_lim
            )
            fdr_mask_L_lim_packed = torch.nn.functional.fold(
                fdr_mask_unfolded.mT.half(),
                output_size=(1, Y.size(-1)),
                kernel_size=(1, L_lim),
                dilation=1,
                padding=0,
                stride=(1, 1),
            ).bool()
            fdr_mask_L_lim = fdr_mask_L_lim_packed.reshape_as(Y)
            # If the subband didn't already contain a FDR
            subband_already_contains_fdr = full_mask.any(dim=-1, keepdim=True).expand_as(full_mask)
            # print(L_lim, subband_already_contains_fdr[0, 0, 240, :].any())
            # Merge it with the other masks
            mask_add = fdr_mask_L_lim.logical_and(subband_already_contains_fdr.logical_not())
            # print(L_lim, mask_add[0, 0, 240, :].any())
            full_mask.logical_or_(mask_add)
            # print(L_lim, full_mask[0, 0, 240, :].any())
            # Early stopping condition of the for loop: when all subbands contain at least one True
            if full_mask.any(dim=-1).all():
                break
        return full_mask

    def compute_individual_masks(self, Y):
        masks = {}
        seen_subbands = set()
        for L_lim in range(self.orig_L_lim, self.minimal_L_lim - 1, -1):
            # Unfold to get candidate sliding windows
            Y_unfolded = Y.unfold(-1, L_lim, 1)
            # Check if the energy is decreasing in the sliding windowed STFT
            candidate_start_positions_tensor = (Y_unfolded.diff(dim=-1) < 0).all(dim=-1).nonzero()
            masks[L_lim] = dict()
            candidate_start_positions = set(tuple(row) for row in candidate_start_positions_tensor.tolist())

            new_start_positions = set(
                position for position in candidate_start_positions if position[:-1] not in seen_subbands
            )
            masks[L_lim] = new_start_positions

            seen_subbands |= set(position[:-1] for position in new_start_positions)
        return masks

    def apply_mask(self, Y, fdr_masks):
        Y_decreasing = {k: [] for k in (itertools.product(range(Y.shape[0]), fdr_masks.keys()))}
        for mask_length, m_set in fdr_masks.items():
            for position in m_set:
                subband = Y[position[:-1]]
                Y_decreasing[(position[0], mask_length)].append(subband[position[-1] : position[-1] + mask_length])
        return {k: torch.stack(v, dim=0) for k, v in Y_decreasing.items() if len(v) > 0}

    def compute_individual_rt_60(self, power):
        edc = torch.flip(torch.cumsum(torch.flip(power, (-1,)), -1), (-1,))
        edc_db = energy_to_db(edc)
        edc_db_scaled = edc_db - edc_db[..., 0, None]
        selected_edc_db_scaled = edc_db_scaled[edc_db_scaled[:, -1] < -abs(self.minimal_db_decrease)]
        indexes_for_regression = arange_last_dim_like(selected_edc_db_scaled)
        if self.edc_regression_intersect_zero:
            slope = (indexes_for_regression * selected_edc_db_scaled).mean(dim=-1) / (indexes_for_regression**2).mean(
                dim=-1
            )
        else:
            edc_db_scaled_mean = selected_edc_db_scaled.nanmean(dim=-1, keepdims=True)
            indexes_for_regression_mean = indexes_for_regression.nanmean(-1, keepdims=True)
            numerator = (
                (indexes_for_regression - indexes_for_regression_mean) * (selected_edc_db_scaled - edc_db_scaled_mean)
            ).nansum(dim=-1, keepdims=True)
            denominator = ((indexes_for_regression - indexes_for_regression_mean) ** 2).nansum(dim=-1, keepdims=True)
            slope = numerator / denominator
        return -60 / slope * self.stft_step_rate / self.fs

    def aggregate_rt_60(self, individual_rt_60s, batch_size):
        res_per_batch = {}
        for k, v in individual_rt_60s.items():
            if k[0] not in res_per_batch.keys():
                res_per_batch[k[0]] = []
            res_per_batch[k[0]].append(v)
        res_tensor = torch.full((batch_size, 1, 1), torch.nan, device=self.polynomial.coeffs.device)
        for idx_in_batch, v in res_per_batch.items():
            res_tensor[idx_in_batch] = torch.cat(v).median()
        # torch.stack(list(res_per_batch.values()), dim=0) # doesn't account for batches where nothing is found
        return res_tensor
        # return median_rt_60

    @classmethod
    def simple_linear_regression(self, x, y, force_origin=False):
        if force_origin:
            return torch.mean(x * y) / torch.mean(x**2), 0
        cov = torch.cov(torch.stack((x, y)))[1, 0]
        # Compute the variance of x
        x_var = torch.var(x)
        # Compute the slope and intercept of the regression line
        slope = cov / x_var
        intercept = y.mean() - slope * x.mean()
        return slope, intercept

    def fit(self, y, rt_60_tgt, savefig_path=None):
        # fits one (big) batch of data
        assert y.ndim == 3 and y.size(0) > self.polynomial.deg
        self.clamp = True
        with torch.no_grad():
            rt_60_pred = self.forward(y)
            self.polynomial.fit(rt_60_pred, rt_60_tgt, solver="lstsq", flatten=True, inplace=True)
            rt_60_corrected = self(y).cpu().squeeze()
            rt_60_tgt = rt_60_tgt.cpu().squeeze()
        # print(f"Best polynomial {self.polynomial}")
        print(f"Best polynomial Coefficients tuple: {tuple(self.polynomial.coeffs.cpu().numpy())}")
        print(f"MSE {nn.MSELoss()(rt_60_corrected, rt_60_tgt).cpu().item():.4f}")
        print(f"MAE {nn.L1Loss()(rt_60_corrected, rt_60_tgt).cpu().item():.4f}")
        print(
            f"Pearson correlation: {scipy.stats.pearsonr(rt_60_corrected.cpu().numpy(), rt_60_tgt.cpu().numpy()).statistic:.4f}"
        )
        print()
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal")
        ax.set_title(
            r"Fitted $RT_{60}$ estimator "
            + f"regression degree = {self.polynomial.deg}, EDC regression intersect zero={self.edc_regression_intersect_zero}"
        )
        # plt.plot(targets, preds, ".")
        ax.plot(rt_60_tgt.sort().values, rt_60_corrected[rt_60_tgt.sort().indices], ".")
        graph_lim = max(rt_60_tgt) + 0.2
        ax.set_xlim(0.0, graph_lim)
        ax.set_ylim(0.0, graph_lim)
        ax.plot([0, graph_lim], [0, graph_lim], "--", color="tab:red")
        ax.set_xlabel("Ground Truth " + "${RT}_{60}$")
        ax.set_ylabel("Estimated ${RT}_{60}$ (after correction)")
        if savefig_path is not None:
            fig.savefig(savefig_path + "/rt_60_tuning.pdf", bbox_inches="tight")
        try:
            fig.show()
        except:
            pass
        self.clamp = False

    def correct_rt_60(self, rt_60):
        res_after_correction = self.polynomial(rt_60)
        res_after_correction[res_after_correction.isnan()] = self.default_rt_60
        if getattr(self, "clamp", True):
            return res_after_correction.clamp(min=0.01)
        else:
            return res_after_correction

    def plot_fdr_mask(self, y):
        Y = self.stft_module(y)
        Y = Y[..., : self.num_cropped_bins, :]
        plt.figure()
        plt.imshow(Y[0].squeeze().log(), origin="lower", cmap="Blues")
        mask = self.compute_fdr_mask(Y)
        plt.imshow(mask[0].squeeze(), origin="lower", cmap="Reds", alpha=0.3)

    def forward(self, y):
        if y.size(-2) != 1:
            raise NotImplementedError("Only 1-channel signals are supported")
        Y = self.stft_module(y)
        Y = Y[..., : self.num_cropped_bins, :]
        fdr_masks = self.compute_individual_masks(Y)
        Y_decreasing = self.apply_mask(Y, fdr_masks)
        individual_rt_60s = {k: self.compute_individual_rt_60(v) for k, v in Y_decreasing.items()}
        rt_60 = self.aggregate_rt_60(individual_rt_60s, batch_size=y.size(0))
        corrected_rt_60 = self.correct_rt_60(rt_60)
        return corrected_rt_60


def test_prego():
    from lightning.pytorch import seed_everything

    seed_everything(12)
    from datasets import (
        WSJSimulatedRirDataModule,
        SynthethicRirDataset,
        WSJDataset,
    )
    from model.reverb_models.early_echoes import MeanFreePathEarlyEnd
    from model.reverb_models.polack import PolackAnalysis

    plt.close("all")
    # fig_ext, ax_ext = plt.subplots()

    rt_60_est = RT60Prego()
    oracle_rt_60_est = PolackAnalysis()

    # torch.manual_seed(0)
    # rir_dataset = SynthethicRirDataset(query="rt_60>0.8")
    rir_dataset = SynthethicRirDataset()
    WSJDataset("data/speech", "test")
    # rir_dataset = SynthethicRirDataset(query='rt_60 > 0.5 and rt_60 < 0.6')
    # rir_dataset = SynthethicRirDataset(query="rt_60 > 0. and rt_60 < 0.3")
    data_module = WSJSimulatedRirDataModule(rir_dataset=rir_dataset, dry_signal_target_len=49151)
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()

    y, (x, h, rir_properties) = next(iter(loader))
    # Y = rt_60_est.stft_module(y)
    # individual_masks = rt_60_est.compute_individual_masks(Y)
    # rt_60_est.plot_fdr_mask(y)
    print("oracle", oracle_rt_60_est(h)[1].squeeze())
    with torch.no_grad():
        print("estimated", rt_60_est(y))


def plot_prego_no_fit(data_module, num_samples=100, edc_regression_intersect_zero: bool = True):
    from lightning.pytorch import seed_everything

    seed_everything(12)
    from datasets import (
        WSJ1SimulatedRirDataModule,
        SynthethicRirDataset,
        WSJDataset,
        EARSReverbDataModule,
    )
    from model.reverb_models.polack import PolackAnalysisRT60Only

    data_module.hparams.batch_size = num_samples

    device = torch.device("cuda")

    rt_60_estimator = RT60Prego(
        fs=16000, polynomial_deg=1, edc_regression_intersect_zero=edc_regression_intersect_zero
    ).to(device=device)

    if data_module is None:
        rir_dataset = SynthethicRirDataset(rir_root="./data/rirs_v2")
        data_module = WSJ1SimulatedRirDataModule(
            rir_dataset=rir_dataset,
            dry_signal_target_len=49151,
            batch_size=10,
            ignore_silent_windows=True,
        )
        oracle_rt_60_est = PolackAnalysisRT60Only(fs=data_module.hparams.fs).to(device=device)
    else:
        if getattr(data_module.hparams, "resample_rir", True):
            oracle_rt_60_est = PolackAnalysisRT60Only(fs=16000, rir_length=64000).to(device=device)
        else:
            oracle_rt_60_est = PolackAnalysisRT60Only(fs=48000, rir_length=192000).to(device=device)
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()
    (y, (x, h, rir_properties)) = next(iter(loader))
    rt_60_tgt = oracle_rt_60_est(h.to(device=device)).squeeze()
    rt_60_prego = rt_60_estimator(y.to(device=device)).squeeze()

    fig, ax = plt.subplots(1, 1)
    ax.set_title(r"Prego $RT_{60}$ estimator")
    # plot estimate
    ax.plot(rt_60_tgt.cpu().numpy(), rt_60_prego.cpu().numpy(), ".")
    # plot fit
    prego_axis = torch.linspace(rt_60_prego.min(), rt_60_prego.max(), 1000)
    for deg in range(1, 7):
        rt_60_estimator.polynomial = Polynomial(deg=deg)
        rt_60_estimator.polynomial.fit(rt_60_prego, rt_60_tgt, inplace=True)
        fitted_curve = rt_60_estimator.polynomial(prego_axis.to(device=device))

        rt_60_corrected = rt_60_estimator.polynomial(rt_60_prego).squeeze()
        print(f"deg={deg}")
        print(f"MSE {nn.MSELoss()(rt_60_corrected, rt_60_tgt).cpu().item():.4f}")
        print(f"MAE {nn.L1Loss()(rt_60_corrected, rt_60_tgt).cpu().item():.4f}")
        print(
            f"Pearson correlation: {scipy.stats.pearsonr(rt_60_corrected.cpu().numpy(), rt_60_tgt.cpu().numpy()).statistic:.4f}"
        )
        print()
        ax.plot(fitted_curve.squeeze().cpu().numpy(), prego_axis.squeeze().cpu().numpy(), label=f"deg={deg}")
    print("From these statistics we see that no improvement after deg 2")
    ax.set_xlabel("Ground Truth " + "${RT}_{60}$")
    ax.set_ylabel("Estimated ${RT}_{60}$ (No correction)")
    ax.legend()
    plt.show()


def fit_prego(data_module, num_samples=100, polynomial_deg=1, edc_regression_intersect_zero: bool = True):
    from lightning.pytorch import seed_everything

    seed_everything(12)
    from datasets import (
        WSJ1SimulatedRirDataModule,
        SynthethicRirDataset,
        WSJDataset,
        EARSReverbDataModule,
    )
    from model.reverb_models.polack import PolackAnalysisRT60Only

    data_module.hparams.batch_size = num_samples

    device = torch.device("cuda")

    rt_60_est = RT60Prego(
        fs=16000, polynomial_deg=polynomial_deg, edc_regression_intersect_zero=edc_regression_intersect_zero
    ).to(device=device)

    if data_module is None:
        rir_dataset = SynthethicRirDataset(rir_root="./data/rirs_v2")
        data_module = WSJ1SimulatedRirDataModule(
            rir_dataset=rir_dataset,
            dry_signal_target_len=49151,
            batch_size=10,
            ignore_silent_windows=True,
        )
        oracle_rt_60_est = PolackAnalysisRT60Only(fs=data_module.hparams.fs).to(device=device)
    else:
        if getattr(data_module.hparams, "resample_rir", True):
            oracle_rt_60_est = PolackAnalysisRT60Only(fs=16000, rir_length=64000).to(device=device)
        else:
            oracle_rt_60_est = PolackAnalysisRT60Only(fs=48000, rir_length=192000).to(device=device)
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()
    (y, (x, h, rir_properties)) = next(iter(loader))
    rt_60_tgt = oracle_rt_60_est(h.to(device=device))
    rt_60_est.fit(y.to(device=device), rt_60_tgt)


if __name__ == "__main__":
    # test_prego()
    from datasets import EARSReverbDataModule

    data_module = EARSReverbDataModule(batch_size=10, return_rir=True, resample_rir=True, enable_caching_val=False)
    plot_prego_no_fit(data_module=data_module, num_samples=500, edc_regression_intersect_zero=False)
    fit_prego(data_module=data_module, num_samples=500, polynomial_deg=1, edc_regression_intersect_zero=True)
    fit_prego(data_module=data_module, num_samples=500, polynomial_deg=1, edc_regression_intersect_zero=False)
    # fit_prego(data_module=data_module, num_samples=500, polynomial_deg=2, edc_regression_intersect_zero=True)
    fit_prego(data_module=data_module, num_samples=500, polynomial_deg=2, edc_regression_intersect_zero=False)
