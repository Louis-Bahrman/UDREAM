#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:41:33 2024

@author: louis
"""

from model.utils.abs_models import AbsReverbModel, OracleParametersReverbModel
from model.utils.tensor_ops import energy_to_db, db_to_energy, Filterbank, tuple_to_device, crop_or_zero_pad_to_target_len
import torch
import torchaudio
from torch import nn
import math
import warnings
from model.reverb_models.rt_60_estimators import RT60Estimator
from model.reverb_models.early_echoes import FixedTimeEarlyEnd
from model.reverb_models.drr import DRR, DRREstimator
import matplotlib.pyplot as plt

# %% Anaysis


def edc(rir: torch.Tensor):
    """
    Energy decay curve.

    Parameters
    ----------
    rir : torch.Tensor
        RIR. Peak should be at index 0.

    Returns
    -------
    torch.Tensor
        Energy Decay Curve.

    """
    power = rir.abs().square()
    return torch.flip(torch.cumsum(torch.flip(power, (-1,)), -1), (-1,))


def tau_from_rt_60(rt_60_in_seconds: float | torch.Tensor, fs: int = 1):
    """Compute energy decay tau from RT60."""
    # fs=1 for tau in seconds
    return rt_60_in_seconds * fs / (3 * math.log(10))


def sigma_from_polack_energy(polack_energy, rt_60, integral_lower_bound=0.0, integral_upper_bound=math.inf, fs=1):
    # rt_60 needs to have the same unit (samples or seconds) as upper and lower bounds for it to work
    tau = tau_from_rt_60(rt_60, fs=fs)
    return torch.sqrt(
        polack_energy
        * 2
        / tau
        / (torch.exp(-2 * integral_lower_bound / tau) - torch.exp(-2 * integral_upper_bound / tau))
    )


def polack_energy(sigma, rt_60, integral_lower_bound=0.0, integral_upper_bound=math.inf, fs=1):
    # rt_60 needs to have the same unit (samples or seconds) as upper and lower bounds for it to work
    tau = tau_from_rt_60(rt_60, fs=fs)
    return (
        sigma**2 * tau / 2 * (torch.exp(-2 * integral_lower_bound / tau) - torch.exp(-2 * integral_upper_bound / tau))
    )


# def simple_linear_regression(x, y, force_origin=False):
#     if force_origin:
#         return torch.mean(x * y) / torch.mean(x**2), 0
#     cov = torch.cov(torch.stack((x, y)))[1, 0]
#     # Compute the variance of x
#     x_var = torch.var(x)
#     # Compute the slope and intercept of the regression line
#     slope = cov / x_var
#     intercept = y.mean() - slope * x.mean()
#     return slope, intercept


# def solve_edc_60dB(edc_db_scaled, force_origin=False, return_linear_approx=False, min_energy=-70.0, max_energy=0.0):
#     if force_origin:
#         indexes_for_regression = torch.where(edc_db_scaled > min_energy)[0]
#     else:
#         indexes_for_regression = torch.where(
#             torch.logical_and(edc_db_scaled <= max_energy, edc_db_scaled > min_energy)
#         )[0]
#     breakpoint()
#     y_for_regression = edc_db_scaled[indexes_for_regression]
#     x_for_regression = indexes_for_regression.float()
#     slope, intercept = simple_linear_regression(x_for_regression, y_for_regression, force_origin=force_origin)
#     rt_60 = -(intercept + 60) / slope
#     if return_linear_approx:
#         return rt_60, (x_for_regression, slope * x_for_regression + intercept)
#     return rt_60


class PolackAnalysis(nn.Module):
    def __init__(
        self,
        regress: bool = True,  # Always True, also makes it differentiable and batched
        regression_max_energy: float = -5.0,
        regression_min_energy: float = -25.0,
        rir_length: int = 16383,
        fs: int = 16000,
        intersect_zero: bool = False,
        rt_60_depends_from_slope_only: bool = True,  # whether to substract the intersect in the computation of the RT60. Should be true for synthesis using Polack's model
        direct_path_duration_ms: float = 2.5,
    ):
        super().__init__()
        if not regress:
            raise NotImplementedError()
        self.regression_min_energy = regression_min_energy
        self.regression_max_energy = regression_max_energy
        self.intersect_zero = intersect_zero
        self.rt_60_depends_from_slope_only = rt_60_depends_from_slope_only
        self.fs = fs
        self.direct_path_duration_ms = direct_path_duration_ms
        self.direct_path_duration_seconds = self.direct_path_duration_ms / 1000
        self.direct_path_duration_samples = round(self.direct_path_duration_seconds * self.fs)

        self.register_buffer("orig_indexes_for_regression", torch.arange(rir_length) / self.fs)

    def forward(self, h):
        h=crop_or_zero_pad_to_target_len(h, self.orig_indexes_for_regression.size(-1))
        edc_h = edc(h)  # + self.epsilon
        edc_db = energy_to_db(edc_h)
        total_energy = edc_db[..., 0, None]
        edc_db_scaled = edc_db - total_energy
        begin_polack_in_samples = torch.argmax(1 * (edc_db_scaled <= self.regression_max_energy), dim=-1, keepdim=True)
        end_polack_in_samples = torch.argmin(1 * (edc_db_scaled > self.regression_min_energy), dim=-1, keepdim=True)
        energy_mask = torch.logical_and(
            edc_db_scaled <= self.regression_max_energy, edc_db_scaled > self.regression_min_energy
        )
        # Clip the mask
        energy_mask[..., : self.direct_path_duration_samples + 1] = False
        begin_polack_in_samples.clamp_(min=self.direct_path_duration_samples + 1)

        edc_db_scaled[~energy_mask] = torch.nan
        indexes_for_regression = self.orig_indexes_for_regression[(h.ndim - 1) * (None,)].repeat(
            *h.shape[:-1], 1
        )  # repeat also copies so it works
        indexes_for_regression[~energy_mask] = torch.nan
        if self.intersect_zero:
            numerator = (indexes_for_regression * edc_db_scaled).nanmean(dim=-1, keepdims=True)
            denominator = (indexes_for_regression**2).nanmean(dim=-1, keepdims=True)
            slope = numerator / denominator
            rt_60 = -60 / slope
        else:
            edc_db_scaled_mean = edc_db_scaled.nanmean(dim=-1, keepdims=True)
            indexes_for_regression_mean = indexes_for_regression.nanmean(-1, keepdims=True)
            numerator = (
                (indexes_for_regression - indexes_for_regression_mean) * (edc_db_scaled - edc_db_scaled_mean)
            ).nansum(dim=-1, keepdims=True)
            denominator = ((indexes_for_regression - indexes_for_regression_mean) ** 2).nansum(dim=-1, keepdims=True)
            slope = numerator / denominator
            intercept = edc_db_scaled_mean - slope * indexes_for_regression_mean
            # rt_60 = -60 / (slope * self.fs)
            if self.rt_60_depends_from_slope_only:
                rt_60 = -60 / slope
            else:
                db_regress_init = (self.regression_max_energy - intercept) / slope
                db_regress_end = (self.regression_min_energy - intercept) / slope
                rt_60 = (
                    -60 / (self.regression_min_energy - self.regression_max_energy) * (db_regress_end - db_regress_init)
                )
        if getattr(self, "plot", False):
            import matplotlib.pyplot as plt

            if self.intersect_zero:
                intercept = torch.zeros_like(slope)
            edc_db_scaled_full = edc_db - total_energy
            num_axes = edc_db_scaled_full.shape[-2] if edc_db_scaled_full.shape[-2] < 32 else 10
            fig, axs = plt.subplots(
                nrows=num_axes,
                ncols=1,
                squeeze=False,
                figsize=(6, 4 * num_axes),
            )
            for i_axis in range(num_axes):
                ax = axs[axs.shape[0] - 1 - i_axis, 0]
                channel_or_band = round(i_axis / num_axes * edc_db_scaled_full.shape[-2])
                # channel or band depends on whether it is scaled or not
                ax.plot(
                    self.orig_indexes_for_regression,
                    edc_db_scaled_full[((0,) * (edc_db_scaled_full.ndim - 2) + (channel_or_band,))],
                    color="tab:blue",
                )
                indexes_to_plot = indexes_for_regression[
                    ((0,) * (indexes_for_regression.ndim - 2) + (channel_or_band,))
                ]
                slope_for_ax = slope[(0,) * (slope.ndim - 2) + (channel_or_band,)]
                intercept_for_ax = intercept[(0,) * (intercept.ndim - 2) + (channel_or_band,)]
                regression_line = slope_for_ax * indexes_to_plot + intercept_for_ax
                regression_line_full = slope_for_ax * self.orig_indexes_for_regression + intercept_for_ax
                ax.plot(indexes_to_plot, regression_line, color="tab:orange")
                ax.plot(
                    self.orig_indexes_for_regression,
                    regression_line_full,
                    "--",
                    color="tab:orange",
                )
                rt_60_for_plot = float(rt_60[(0,) * (rt_60.ndim - 2) + (channel_or_band,)])
                ax.vlines(rt_60_for_plot, -80, 1, label=f"RT60 = {rt_60_for_plot:.2f} s")
                ax.set_ylim(-80, 1)
                ax.set_xlabel("time")
                ax.set_ylabel("Energy (dB)")
                channel_or_band_str = "channel" if edc_db_scaled_full.ndim == 3 else "band"
                ax.set_title(
                    f"EDC (dB) of {channel_or_band_str} {channel_or_band}, (intercept in 0={self.intersect_zero})"
                )
                ax.legend()
            fig.tight_layout()
            # fig.show()
        # Actually, valid_energy == self.regression_max_energy - regression_min_energy but in dB
        # energy_to_db(-h_valid_energy.square().nansum(dim=-1,keepdim=True)+h_invalid_energy.square().nansum(dim=-1,keepdim=True)) - total_energy
        # valid_energy = 10 ** ((self.regression_max_energy - self.regression_min_energy) / 10)
        valid_energy = edc_h.gather(dim=-1, index=begin_polack_in_samples) - edc_h.gather(
            dim=-1, index=end_polack_in_samples
        )
        # we need to compute sigma in samples because the valid_energy is computed in Power \times samples
        sigma = sigma_from_polack_energy(
            valid_energy,
            rt_60=rt_60,
            integral_lower_bound=begin_polack_in_samples,
            integral_upper_bound=end_polack_in_samples,
            fs=self.fs,
        )
        return sigma, rt_60


class DirectToPolackRatio(DRR):
    def __init__(
        self,
        polack_analysis: nn.Module | None = None,
        fs: int = 16000,
        ms_after_peak: float = 2.5,
        assume_peak_at_beginning=True,
    ):
        super().__init__(
            fs=fs,
            ms_after_peak=ms_after_peak,
            assume_peak_at_beginning=assume_peak_at_beginning,
        )
        self.polack_analysis = polack_analysis

    def forward(self, h, sigma=None, rt_60=None):
        if sigma is None or rt_60 is None:
            analyzed_sigma, analyzed_rt_60 = self.polack_analysis(h)
            if sigma is None:
                sigma = analyzed_sigma
            if rt_60 is None:
                rt_60 = analyzed_rt_60
        estimated_polack_energy = polack_energy(
            sigma, rt_60, integral_lower_bound=self.num_samples_after_peak, integral_upper_bound=math.inf, fs=self.fs
        )
        direct_energy = self.direct_energy(h)
        return energy_to_db(direct_energy / estimated_polack_energy)


# %% Synthesis


class ReverberationTimeShortening(nn.Module):
    def __init__(self, sr=16000, time_after_max=0.0025, assume_peak_at_beginning: bool = True):
        super().__init__()
        self.sr = sr
        self.time_after_max = time_after_max
        if not assume_peak_at_beginning:
            raise NotImplementedError()

    def forward(self, rir, original_T60, target_T60):
        """Shorten reverberation time of a RIR.

        See this paper for more details:
            Speech Dereverberation With a Reverberation Time Shortening Target
            https://arxiv.org/abs/2204.08765

        Args:
            rir: given RIR.
            original_T60: the rt60 of the given RIR.
            target_T60: the target rt60.
            sr: sampling rate. Defaults to 16000.
            time_after_max: the time after the maximum of the RIR. Defaults to 0.002.

        Returns:
            The shortened RIR and the window.

        Cite:
            @article{zhou2022single,
                title={Single-Channel Speech Dereverberation using Subband Network with A Reverberation Time Shortening Target},
                author={Zhou, Rui and Zhu, Wenye and Li, Xiaofei},
                journal={arXiv preprint arXiv:2204.08765},
                year={2022}
            }
        """
        assert rir.squeeze().ndim == 1, "rir must be a 1D array."

        q = 3 / (target_T60 * self.sr) - 3 / (original_T60 * self.sr)
        idx_max = 0
        N1 = int(idx_max + self.time_after_max * self.sr)
        win = torch.empty_like(rir)
        win[..., :N1] = 1
        win[..., N1:] = 10 ** (-q * torch.arange(rir.shape[-1] - N1))
        rir_shortened = rir * win
        # return rir , win
        return rir_shortened


class PolackSynthesis(nn.Module):
    def __init__(
        self,
        early_echoes_masking_module: nn.Module,
        rir_length: int = 16383,
        fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
    ):
        super().__init__()
        self.early_echoes_masking_module = early_echoes_masking_module
        self.rir_length = rir_length
        self.fs = fs
        self.register_buffer("T", torch.arange(self.rir_length) / self.fs)
        self.positive_valued = positive_valued
        self.num_polack_draws = num_polack_draws
        self.fixed_sigma = fixed_sigma

    def expand_v_for_polack_draws(self, v):
        return v.expand(*v.shape[:-2], self.num_polack_draws, -1)

    def compute_v(self, sigma, rt_60, early_reverb_mask):
        if isinstance(early_reverb_mask, torch.Tensor) and early_reverb_mask.ndim == 1:
            early_reverb_mask = early_reverb_mask[(sigma.ndim - 1) * (None,)].expand(*sigma.shape[:-1], -1)
        tau = tau_from_rt_60(rt_60)
        if self.fixed_sigma is not None:
            sigma = self.fixed_sigma
        T_divided_by_tau = self.T / tau
        v = sigma * torch.exp(-T_divided_by_tau)
        v[early_reverb_mask] = 0
        v[~v.isfinite()] = 0  # In case sigma and tau wrongly estimated (because of early echoes and very short rir)
        v = self.expand_v_for_polack_draws(v)
        # ax_ext.plot(v[0].squeeze(), label="variance")
        return v

    def forward(self, sigma, rt_60, rir_properties=dict(), peak_amplitude=1.0):
        early_reverb_mask = self.early_echoes_masking_module.compute_early_mask(rir_properties)
        v = self.compute_v(sigma, rt_60, early_reverb_mask=early_reverb_mask)
        rir = torch.randn_like(v) * v
        rir[..., 0] = peak_amplitude
        if self.positive_valued:
            rir = rir.abs()
        return rir


class BandwisePolackSynthesis(PolackSynthesis):
    def __init__(
        self,
        early_echoes_masking_module: nn.Module,
        num_evenly_spaced_filters: int,
        filter_order: int = 2,
        rir_length: int = 16383,
        fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
    ):
        super().__init__(
            early_echoes_masking_module=early_echoes_masking_module,
            rir_length=rir_length,
            fs=fs,
            positive_valued=positive_valued,
            num_polack_draws=num_polack_draws,
            fixed_sigma=fixed_sigma,
        )
        self.filterbank = Filterbank(num_evenly_spaced_filters=num_evenly_spaced_filters, filter_order=filter_order)

    def expand_v_for_polack_draws(self, v):
        return v.expand(*v.shape[:-3], self.num_polack_draws, -1, -1)

    def forward(self, sigma, rt_60, rir_properties, peak_amplitude=1.0):
        early_reverb_mask = self.early_echoes_masking_module.compute_early_mask(rir_properties)
        v = self.compute_v(sigma, rt_60, early_reverb_mask=early_reverb_mask)
        white_noise = torch.randn_like(v[..., 0, :])
        filtered_white_noise = self.filterbank(white_noise)
        rir = filtered_white_noise * v
        rir[..., 0] = peak_amplitude
        rir = self.filterbank.inverse(rir)
        if self.positive_valued:
            rir = rir.abs()
        return rir


class STFTPolackSynthesis(PolackSynthesis):
    def __init__(
        self,
        early_echoes_masking_module: nn.Module,
        n_fft: int,
        hop_length: int,
        window_fn,
        use_GL: bool = False,  # wether to use Griffin Lim or simply a random phase
        rir_length: int = 16383,
        fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
        num_GL_iterations: int | None = 32,
        GL_momentum: float | None = 0.99,
    ):

        if hop_length is None:
            hop_length = n_fft // 2

        rir_num_frames = rir_length // hop_length + 1
        super().__init__(
            early_echoes_masking_module=early_echoes_masking_module,
            rir_length=rir_num_frames,
            fs=fs / hop_length,
            positive_valued=positive_valued,
            num_polack_draws=num_polack_draws,
            fixed_sigma=fixed_sigma,
        )
        self.use_GL = use_GL
        self.griffin_lim_module = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=num_GL_iterations,
            hop_length=hop_length,
            window_fn=window_fn,
            power=2.0,
            length=rir_length,
            momentum=GL_momentum,
            rand_init=True,
        )
        self.istft_module = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop_length, window_fn=window_fn
        )

    def expand_v_for_polack_draws(self, v):
        return v.expand(*v.shape[:-3], self.num_polack_draws, -1, -1)

    def forward(self, sigma, rt_60, rir_properties, peak_amplitude=1.0):
        # convert rt_60 to frames
        rt_60_in_frames = rt_60  # / self.griffin_lim_module.hop_length
        # breakpoint()
        # compute STFT, ignore mask for phase reconstruction
        v = self.compute_v(sigma, rt_60_in_frames, early_reverb_mask=False)
        if getattr(self, "plot", False):
            plt.figure()
            plt.imshow(v[(0,) * (v.ndim - 2)].log10() * 10, origin="lower")
        if self.use_GL:
            # reconstruct phase using GL
            rir = self.griffin_lim_module(v)
        else:
            args = torch.rand_like(v)
            phases = torch.exp(2j * torch.pi * args)
            complex_spectrogram = v.sqrt() * phases
            rir = self.istft_module(complex_spectrogram, length=self.early_echoes_masking_module.rir_length)
        rir = self.early_echoes_masking_module.rir_to_late(rir, rir_properties)
        rir[..., 0] = peak_amplitude
        if self.positive_valued:
            rir = rir.abs()
        return rir


# %% Analysis-synthesis


class RirToPolack(torch.nn.Module):
    def __init__(
        self,
        early_echoes_masking_module: torch.nn.Module,
        regress: bool = True,  # also makes it differentiable and batched
        regression_max_energy: float = -5.0,
        regression_min_energy: float = -25.0,
        analysis_rir_length: int = 16383,
        synthesis_rir_length: int = 16383,
        intersect_zero: bool = False,
        rt_60_depends_from_slope_only: bool = True,  # whether to substract the intersect in the computation of the RT60. Should be true for synthesis using Polack's model
        analysis_normalization_method: str | None = None,
        synthesis_normalization_method: str | None = None,
        analysis_fs: int = 16000,
        synthesis_fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
        move_all_direct_energy_to_peak: bool = False,  # should be true for blind
        direct_path_duration_ms=2.5,
    ):
        super().__init__()
        self.oracle_polack_analysis = PolackAnalysis(
            regress=regress,
            regression_max_energy=regression_max_energy,
            regression_min_energy=regression_min_energy,
            rir_length=analysis_rir_length,
            fs=analysis_fs,
            intersect_zero=intersect_zero,
            rt_60_depends_from_slope_only=rt_60_depends_from_slope_only,
            direct_path_duration_ms=direct_path_duration_ms,
        )
        self.polack_synthesis = PolackSynthesis(
            early_echoes_masking_module=early_echoes_masking_module,
            rir_length=synthesis_rir_length,
            fs=synthesis_fs,
            positive_valued=positive_valued,
            num_polack_draws=num_polack_draws,
            fixed_sigma=fixed_sigma,
        )
        self.analysis_normalization_method = analysis_normalization_method
        self.synthesis_normalization_method = synthesis_normalization_method
        self.oracle_drr_module_for_analysis = DirectToPolackRatio(
            None, fs=analysis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.oracle_drr_module_for_synthesis = DirectToPolackRatio(
            None, fs=synthesis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.move_all_direct_energy_to_peak = move_all_direct_energy_to_peak

    def normalize_rir(self, h, stage):
        # Normalize h
        if stage == "analysis":
            normalization_method = self.analysis_normalization_method
            drr_module = self.oracle_drr_module_for_analysis
        else:
            normalization_method = self.synthesis_normalization_method
            drr_module = self.oracle_drr_module_for_synthesis

        if normalization_method is None or normalization_method == "" or "none" in normalization_method.lower():
            return h
        if "peak" in normalization_method.lower():
            return h / (h[..., 0].abs().unsqueeze(-1))
        if "direct" in normalization_method.lower() and "energy" in normalization_method.lower():
            return h / drr_module.direct_energy(h).sqrt()
        if "total" in normalization_method.lower() and "energy" in normalization_method.lower():
            return h / h.abs().square().sum(dim=-1, keepdims=True).sqrt()
        if "rms" in normalization_method.lower():
            return h / h.abs().square().mean(dim=-1, keepdims=True).sqrt()
        else:
            raise ValueError()

    def convert_rir(self, h, rir_properties):
        h_normalized = self.normalize_rir(h, stage="analysis")
        if getattr(self, "plot", False):
            fig, ax = plt.subplots(1, 1)
            ax.plot(
                self.oracle_polack_analysis.orig_indexes_for_regression.cpu().numpy(),
                h_normalized[0].squeeze().detach().cpu().numpy(),
                label="Target RIR (normalized)",
            )

        normalized_peak_from_analysis = h_normalized[..., 0].abs()
        sigma, rt_60 = self.oracle_polack_analysis(h_normalized)
        target_drr_db = self.oracle_drr_module_for_analysis(h_normalized, sigma=sigma, rt_60=rt_60)
        # convert from dB
        target_drr_linear = db_to_energy(target_drr_db)
        # compute reverberant energy at target fs under polack's model
        target_reverberant_energy_at_target_fs = polack_energy(
            sigma,
            rt_60,
            integral_lower_bound=self.oracle_drr_module_for_analysis.direct_path_duration_seconds,
            fs=self.polack_synthesis.fs,
        )
        # compute the target direct energy
        target_direct_energy_at_target_fs = target_drr_linear * target_reverberant_energy_at_target_fs
        if self.move_all_direct_energy_to_peak:
            peak_for_synthesis = target_direct_energy_at_target_fs.sqrt()[..., 0]
        else:
            peak_for_synthesis = normalized_peak_from_analysis
        h_hat = self.polack_synthesis(sigma, rt_60, rir_properties, peak_for_synthesis)
        h_hat_normalized = self.normalize_rir(h_hat, stage="synthesis")
        # print(
        #     f"DRR: {target_drr_db[0].squeeze().item():.2f} -> {energy_to_db(self.oracle_drr_module_for_synthesis.direct_energy(h_hat_normalized)/self.oracle_drr_module_for_synthesis.reverberant_energy(h_hat_normalized))[0].squeeze().item():.2f}"
        # )
        if getattr(self, "plot", False):
            ax.plot(
                self.polack_synthesis.T.cpu().numpy(),
                h_hat_normalized[(0,) * (h_hat.ndim - 1)].squeeze().cpu().numpy(),
                label="Synthesized RIR (normalized)",
            )
            ax.legend()
        return h_hat_normalized

    def forward(self, h, rir_properties):
        return self.convert_rir(h, rir_properties)


class BandwiseRirToPolack(RirToPolack):
    def __init__(
        self,
        early_echoes_masking_module: torch.nn.Module,
        num_evenly_spaced_synthesis_filters: int,
        filter_order: int = 2,
        regress: bool = True,  # also makes it differentiable and batched
        regression_max_energy: float = -5.0,
        regression_min_energy: float = -25.0,  # change to match bandwise which is less precise it seems
        analysis_rir_length: int = 192000,
        synthesis_rir_length: int = 32000,
        intersect_zero: bool = False,
        rt_60_depends_from_slope_only: bool = True,  # whether to substract the intersect in the computation of the RT60. Should be true for synthesis using Polack's model
        analysis_normalization_method: str | None = "direct energy",
        synthesis_normalization_method: str | None = None,
        analysis_fs: int = 48000,
        synthesis_fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
        move_all_direct_energy_to_peak: bool = True,  # should be true for blind
        direct_path_duration_ms=2.5,
    ):
        super(RirToPolack, self).__init__()
        self.oracle_polack_analysis = PolackAnalysis(
            regress=regress,
            regression_max_energy=regression_max_energy,
            regression_min_energy=regression_min_energy,
            rir_length=analysis_rir_length,
            fs=analysis_fs,
            intersect_zero=intersect_zero,
            rt_60_depends_from_slope_only=rt_60_depends_from_slope_only,
            direct_path_duration_ms=direct_path_duration_ms,
        )
        self.polack_synthesis = BandwisePolackSynthesis(
            early_echoes_masking_module=early_echoes_masking_module,
            rir_length=synthesis_rir_length,
            fs=synthesis_fs,
            positive_valued=positive_valued,
            num_polack_draws=num_polack_draws,
            fixed_sigma=fixed_sigma,
            num_evenly_spaced_filters=num_evenly_spaced_synthesis_filters,
            filter_order=filter_order,
        )
        self.synthesis_filterbank = self.polack_synthesis.filterbank
        self.analysis_normalization_method = analysis_normalization_method
        self.synthesis_normalization_method = synthesis_normalization_method
        self.oracle_drr_module_for_analysis = DirectToPolackRatio(
            None, fs=analysis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.oracle_drr_module_for_synthesis = DirectToPolackRatio(
            None, fs=synthesis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.move_all_direct_energy_to_peak = move_all_direct_energy_to_peak

        self.num_evenly_spaced_synthesis_filters = num_evenly_spaced_synthesis_filters
        if analysis_fs < synthesis_fs:
            raise ValueError("cannot upsample RIR")
        self.num_evenly_spaced_analysis_filters = (
            self.num_evenly_spaced_synthesis_filters * analysis_fs
        ) // synthesis_fs
        self.analysis_filterbank = Filterbank(
            num_evenly_spaced_filters=self.num_evenly_spaced_analysis_filters, filter_order=filter_order, fs=analysis_fs
        )
        if not (
            (analysis_normalization_method is None or "none" in analysis_normalization_method.lower())
            and (synthesis_normalization_method is None or "none" in synthesis_normalization_method.lower())
        ):
            raise NotImplementedError()
        # if not (
        #     "direct" in analysis_normalization_method.lower() and "energy" in analysis_normalization_method.lower()
        # ):
        #     raise NotImplementedError()
        if not move_all_direct_energy_to_peak:
            raise NotImplementedError()

    def convert_rir(self, h, rir_properties):
        h_normalized = self.normalize_rir(h, stage="analysis")
        direct_energy = self.oracle_drr_module_for_analysis.direct_energy(h_normalized)
        # use polack energy for tail
        h_filterbanked = self.analysis_filterbank(h_normalized)
        # we crop to avoid the filters above the synthesis Nyquist frequency
        h_filterbanked = h_filterbanked[..., : self.num_evenly_spaced_synthesis_filters, :]
        h_filterbanked_normalized = h_filterbanked
        # h_filterbanked_normalized = self.normalize_rir(h_filterbanked, stage="analysis")
        if getattr(self, "plot", False):
            fig, (ax_analysis, ax_synthesis) = plt.subplots(1, 2)
            # We shift each band of h to plot it not superposed
            h_filterbanked_for_plot = (
                h_filterbanked_normalized[0].squeeze().T.cpu().numpy()
                + (
                    2
                    * h_filterbanked_normalized[0].squeeze().abs().max().cpu()
                    * torch.arange(h_filterbanked_normalized.size(-2))
                ).numpy()
            )
            ax_analysis.plot(
                self.oracle_polack_analysis.orig_indexes_for_regression.cpu().numpy(),
                h_filterbanked_for_plot,
                label="Target RIR (normalized)",
            )

        normalized_peak_from_analysis = h_filterbanked_normalized[..., 0].abs()
        sigma, rt_60 = self.oracle_polack_analysis(h_filterbanked_normalized)
        sigma = sigma[..., : self.num_evenly_spaced_synthesis_filters, :]
        sigma = math.sqrt(self.num_evenly_spaced_synthesis_filters) * sigma
        rt_60 = rt_60[..., : self.num_evenly_spaced_synthesis_filters, :]
        # remove direct energy that is not on path
        direct_energy = direct_energy / self.oracle_polack_analysis.fs * self.polack_synthesis.fs
        direct_bandwise_energy = direct_energy / self.num_evenly_spaced_synthesis_filters**2

        # target_drr_db = self.oracle_drr_module_for_analysis(h_filterbanked_normalized, sigma=sigma, rt_60=rt_60)
        # # convert from dB
        # target_drr_linear = db_to_energy(target_drr_db)
        # # compute reverberant energy at target fs under polack's model
        # target_reverberant_energy_at_target_fs = polack_energy(
        #     sigma,
        #     rt_60,
        #     integral_lower_bound=self.oracle_drr_module_for_analysis.direct_path_duration_seconds,
        #     fs=self.polack_synthesis.fs,
        # )
        # compute the target direct energy
        # target_direct_energy_at_target_fs = target_drr_linear * target_reverberant_energy_at_target_fs
        if self.move_all_direct_energy_to_peak:
            peak_for_synthesis = direct_bandwise_energy.sqrt()
        else:
            peak_for_synthesis = normalized_peak_from_analysis
        h_hat = self.polack_synthesis(sigma, rt_60, rir_properties, peak_for_synthesis)
        h_hat_normalized = self.normalize_rir(h_hat, stage="synthesis")
        if getattr(self, "plot", False):
            ax_synthesis.plot(
                self.polack_synthesis.T.cpu().numpy(),
                h_hat_normalized[(0,) * (h_hat.ndim - 1)].squeeze().cpu().numpy(),
                label="Synthesized RIR (normalized)",
            )
            ax_synthesis.legend()
            # pass
        return h_hat_normalized


class STFTRIRToPolack(RirToPolack):
    def __init__(
        self,
        early_echoes_masking_module: torch.nn.Module,
        synthesis_n_fft: int,
        use_GL: bool = False,
        synthesis_hop_length: int | None = None,
        num_GL_iterations: int | None = 32,
        GL_momentum: float | None = 0.99,
        regress: bool = True,  # also makes it differentiable and batched
        regression_max_energy: float = -5.0,
        regression_min_energy: float = -25.0,  # change to match bandwise which is less precise it seems
        analysis_rir_length: int = 192000,
        synthesis_rir_length: int = 32000,
        intersect_zero: bool = False,
        rt_60_depends_from_slope_only: bool = True,  # whether to substract the intersect in the computation of the RT60. Should be true for synthesis using Polack's model
        analysis_normalization_method: str | None = None,
        synthesis_normalization_method: str | None = None,
        analysis_fs: int = 48000,
        synthesis_fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
        move_all_direct_energy_to_peak: bool = True,  # should be true for blind
        direct_path_duration_ms=2.5,
    ):
        super(RirToPolack, self).__init__()
        if not use_GL and synthesis_hop_length is None:
            synthesis_hop_length = synthesis_n_fft
        if synthesis_n_fft == synthesis_hop_length:
            window_fn = torch.ones
        else:
            window_fn = torch.hann_window
        self.polack_synthesis = STFTPolackSynthesis(
            early_echoes_masking_module=early_echoes_masking_module,
            n_fft=synthesis_n_fft,
            use_GL=use_GL,
            window_fn=window_fn,
            hop_length=synthesis_hop_length,
            num_GL_iterations=num_GL_iterations,
            GL_momentum=GL_momentum,
            rir_length=synthesis_rir_length,
            fs=synthesis_fs,
            positive_valued=positive_valued,
            num_polack_draws=num_polack_draws,
            fixed_sigma=fixed_sigma,
        )
        self.analysis_fs = analysis_fs
        self.synthesis_fs = synthesis_fs
        analysis_hop_length = round(self.synthesis_hop_length * analysis_fs / synthesis_fs)
        analysis_n_fft = round(self.synthesis_n_fft * analysis_fs / synthesis_fs)
        analysis_rir_num_frames = analysis_rir_length // analysis_hop_length + 1
        analysis_frame_rate = analysis_fs / analysis_hop_length
        self.oracle_polack_analysis = PolackAnalysis(
            regress=regress,
            regression_max_energy=regression_max_energy,
            regression_min_energy=regression_min_energy,
            rir_length=analysis_rir_num_frames,
            fs=analysis_frame_rate,
            intersect_zero=intersect_zero,
            rt_60_depends_from_slope_only=rt_60_depends_from_slope_only,
            direct_path_duration_ms=direct_path_duration_ms,
        )
        self.analysis_stft_module = torchaudio.transforms.Spectrogram(
            n_fft=analysis_n_fft,
            hop_length=analysis_hop_length,
            power=None,
            window_fn=window_fn,
        )
        self.analysis_normalization_method = analysis_normalization_method
        self.synthesis_normalization_method = synthesis_normalization_method
        self.oracle_drr_module_for_analysis = DirectToPolackRatio(
            None, fs=analysis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.oracle_drr_module_for_synthesis = DirectToPolackRatio(
            None, fs=synthesis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.move_all_direct_energy_to_peak = move_all_direct_energy_to_peak

        if analysis_fs < synthesis_fs:
            raise ValueError("cannot upsample RIR")
        if not (
            (analysis_normalization_method is None or "none" in analysis_normalization_method.lower())
            and (synthesis_normalization_method is None or "none" in synthesis_normalization_method.lower())
        ):
            raise NotImplementedError()
        # if not (
        #     "direct" in analysis_normalization_method.lower() and "energy" in analysis_normalization_method.lower()
        # ):
        #     raise NotImplementedError()
        if not move_all_direct_energy_to_peak:
            raise NotImplementedError()

    @property
    def synthesis_n_fft(self):
        return self.polack_synthesis.griffin_lim_module.n_fft

    @property
    def synthesis_hop_length(self):
        return self.polack_synthesis.griffin_lim_module.hop_length

    @property
    def analysis_n_fft(self):
        return self.analysis_stft_module.n_fft

    @property
    def analysis_hop_length(self):
        return self.analysis_stft_module.hop_length

    def convert_rir(self, h, rir_properties):
        h_normalized = self.normalize_rir(h, stage="analysis")
        direct_energy = self.oracle_drr_module_for_analysis.direct_energy(h_normalized)
        # use polack energy for tail
        H = self.analysis_stft_module(h_normalized)
        # we crop to avoid the filters above the synthesis Nyquist frequency
        H = H[..., : self.synthesis_n_fft // 2 + 1, :]
        # h_filterbanked_normalized = self.normalize_rir(h_filterbanked, stage="analysis")
        if getattr(self, "plot", False):
            fig, ((ax_analysis_tf, ax_synthesis_tf), (ax_analysis_time, ax_synthesis_time)) = plt.subplots(
                2, 2, sharex="row", sharey="row"
            )
            H_for_plot = 20 * H.abs().log10()[0].squeeze()
            H_for_plot = (H_for_plot - H_for_plot.max().abs()).clamp(min=-60)
            ax_analysis_tf.imshow(H_for_plot.cpu().numpy(), origin="lower")
            ax_analysis_tf.set_title("Target RIR (normalized)")
            ax_analysis_time.plot(torch.arange(h.size(-1)).numpy() / self.analysis_fs, h[(0,) * (h.ndim - 1)].numpy())

        normalized_peak_from_analysis = H[..., 0].abs()
        sigma, rt_60 = self.oracle_polack_analysis(H)
        # Do some scaling on sigma
        sigma = sigma / math.sqrt(self.synthesis_n_fft)
        # remove direct energy that is not on path
        # direct_energy = direct_energy / self.analysis_fs * self.synthesis_fs
        direct_bandwise_energy = direct_energy
        # direct_bandwise_energy = direct_energy / self.synthesis_n_fft**2
        if self.move_all_direct_energy_to_peak:
            peak_for_synthesis = direct_bandwise_energy.sqrt()[..., 0]
        else:
            peak_for_synthesis = normalized_peak_from_analysis
        h_hat = self.polack_synthesis(sigma, rt_60, rir_properties, peak_for_synthesis)
        h_hat_normalized = self.normalize_rir(h_hat, stage="synthesis")
        if getattr(self, "plot", False):
            H_hat = torch.stft(
                h_hat_normalized[(0,) * (h_hat.ndim - 1)],
                n_fft=self.synthesis_n_fft,
                hop_length=self.synthesis_hop_length,
                window=torch.hann_window(self.synthesis_n_fft, device=h_hat_normalized.device),
                return_complex=True,
            )
            H_hat_for_plot = 20 * H_hat.abs().log10()
            H_hat_for_plot = (H_hat_for_plot - H_hat_for_plot.max().abs()).clamp(min=-60)
            ax_synthesis_tf.imshow(H_hat_for_plot, origin="lower")
            ax_synthesis_tf.set_title("Synthesized RIR (normalized)")
            ax_synthesis_time.plot(
                torch.arange(h_hat.size(-1)).numpy() / self.synthesis_fs, h_hat[(0,) * (h_hat.ndim - 1)].numpy()
            )

        return h_hat_normalized


class OraclePolackAnalysisSynthesis(RirToPolack, OracleParametersReverbModel): ...


def DefaultOraclePolackAnalysisSynthesisSynthethic(OracleBandwisePolackAnalysisSynthesis):
    def __init__(self):
        super().__init__(
            early_echoes_masking_module=FixedTimeEarlyEnd(fs=16000, end_time=0.0025, rir_length=16383),
            regress=True,
            regression_max_energy=-5.0,
            regression_min_energy=-35.0,
            analysis_rir_length=16383,
            synthesis_rir_length=16383,
            intersect_zero=False,
            rt_60_depends_from_slope_only=True,
            analysis_normalization_method=None,
            synthesis_normalization_method="peak",
            analysis_fs=16000,
            synthesis_fs=16000,
            positive_valued=True,
            num_polack_draws=1,
            fixed_sigma=None,
            move_all_direct_energy_to_peak=False,
            direct_path_duration_ms=2.5,
        )


class OracleBandwisePolackAnalysisSynthesis(BandwiseRirToPolack, OracleParametersReverbModel): ...


class OracleSTFTPolackAnalysisSynthesis(STFTRIRToPolack, OracleParametersReverbModel): ...


# %% Only RT_60 extraction


class PolackAnalysisRT60Only(PolackAnalysis):
    def forward(self, h):
        _, rt_60 = super().forward(h)
        return rt_60


class PolackToISMRT60Distance(nn.Module):
    def __init__(
        self,
        rt_60_estimator: nn.Module = PolackAnalysisRT60Only(),
    ):
        super().__init__()
        self.rt_60_estimator = rt_60_estimator

    def forward(self, pred, target):
        pred_rt_60 = self.rt_60_estimator(target[0])
        target_rt_60 = target[1]["rt_60"]
        return (pred_rt_60 - target_rt_60).abs().mean()


# %% Blind


class BlindPolack(AbsReverbModel):
    def __init__(
        self,
        rt_60_estimator: RT60Estimator,
        polack_synthesis: PolackSynthesis,
        blind_drr_estimator: DRREstimator | None = None,
        oracle_polack_analysis: PolackAnalysis | None = None,
        internal_loss_objective: str | None = "drr",
        synthesis_normalization_method: str | None = None,
        num_samples_rt_60_tuning: int | None = 100,
        metrics: list[nn.Module] = [],
    ):
        super().__init__(metrics=metrics, crop_input_to_target=False)
        self.polack_synthesis = polack_synthesis
        self.rt_60_estimator = rt_60_estimator
        self.blind_drr_estimator = blind_drr_estimator
        self.num_samples_rt_60_tuning = num_samples_rt_60_tuning
        self.oracle_polack_analysis = oracle_polack_analysis
        if self.oracle_polack_analysis is not None:
            self.oracle_direct_to_polack_ratio = DirectToPolackRatio(
                polack_analysis=None,
                fs=self.oracle_polack_analysis.fs,
                ms_after_peak=self.oracle_polack_analysis.direct_path_duration_ms,
                assume_peak_at_beginning=True,
            )
        self.synthesis_normalization_method = synthesis_normalization_method
        self.internal_loss_objective = internal_loss_objective
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()

        self.check_options_compatible()

    def check_options_compatible(self):
        # no drr if fixed sigma
        if self.blind_drr_estimator is None and self.oracle_polack_analysis_synthesis.fixed_sigma is None:
            raise ValueError("Need to define sigma")
        # Check normalization_method
        if self.synthesis_normalization_method.lower() not in ("total energy", "total_energy", "peak", "rms"):
            raise NotImplementedError()
        # check internal loss variant
        if self.internal_loss_objective is not None and self.internal_loss_objective.lower() not in (
            "",
            "none",
            "drr",
            "rt_60",
            "rt60",
            "sigma",
        ):
            raise ValueError("Internal loss variant not supported")
        # we need to take grat care that prego is not trained. Put it outside of the training step. Done using
        # Maybe add a parameter since we want rt_60 and DRR to be trained separately ?

    def forward(self, y):
        # Note that h is not used here
        rt_60 = self.rt_60_estimator(y)
        drr = self.blind_drr_estimator(y)
        # we consider that the direct energy is 1
        reverberant_energy_db = -drr
        reverberant_energy_linear = db_to_energy(reverberant_energy_db)
        sigma = sigma_from_polack_energy(
            polack_energy=reverberant_energy_linear,
            rt_60=rt_60,
            integral_lower_bound=self.polack_synthesis.early_echoes_masking_module.end_time,
        )
        h_hat = self.polack_synthesis(sigma, rt_60, rir_properties={})
        h_hat_normalized = self.normalize_rir(h_hat)

        return h_hat_normalized, (drr, sigma, rt_60)

    def internal_loss(self, pred, target):
        # return rt_60 MSE or DRR MSE depending on mode
        h_normalized_pred, (drr_pred, sigma_pred, rt_60_pred) = pred
        h_tgt, rir_properties_tgt = target
        # target_h does not need to be normalized because predicted_sigma and predicted_drr are computed before normalization
        if self.internal_loss_objective is None or self.internal_loss_objective.lower() in ("", "none"):
            return None
        # if the variant is not None than we use the oracle model
        sigma_tgt, rt_60_tgt = self.oracle_polack_analysis(h_tgt)
        drr_tgt = self.oracle_direct_to_polack_ratio(h_tgt, sigma=sigma_tgt, rt_60=rt_60_tgt)
        drr_mse = self.mse_loss(drr_pred, drr_tgt)
        sigma_mse = self.mse_loss(sigma_pred, sigma_tgt)
        rt_60_mse = self.mse_loss(rt_60_pred, rt_60_tgt)

        # cannot use log_for_future_aggregate here because batch_idx not provided but fine
        self.log(self._current_fx_name + "_DRR_MSE", drr_mse, on_epoch=True)
        self.log(self._current_fx_name + "_RT_60_MSE", rt_60_mse, on_epoch=True)
        self.log(self._current_fx_name + "_sigma_MSE", sigma_mse, on_epoch=True)

        if self.internal_loss_objective.lower() == "drr":
            drr_mae = self.mae_loss(drr_pred, drr_tgt)
            self.log(self._current_fx_name + "_DRR_MAE", drr_mae, on_epoch=True)
            return drr_mse
        if self.internal_loss_objective.lower() == "sigma":
            sigma_mae = self.mae_loss(sigma_pred, sigma_tgt)
            self.log(self._current_fx_name + "_sigma_MAE", sigma_mae, on_epoch=True)
            return sigma_mse
        if self.internal_loss_objective.lower() in ("rt_60", "rt60"):
            rt_60_mae = self.mae_loss(rt_60_pred, rt_60_tgt)
            self.log(self._current_fx_name + "_RT_60_MAE", rt_60_mae, on_epoch=True)
            return rt_60_mse

    def get_time(self, pred, *args, **kwargs):
        h_hat_normalized, (drr_pred, sigma_pred, rt_60_pred) = pred
        return h_hat_normalized

    def on_train_start(self):
        if self.num_samples_rt_60_tuning is not None:
            self.tune_rt_60_estimator()

    def tune_rt_60_estimator(self):
        print(f"Tuning RT_60 estimator using 1 big batch of size {self.num_samples_rt_60_tuning}")
        orig_loader_id = id(self.trainer.train_dataloader)
        orig_batch_size = self.trainer.datamodule.hparams.batch_size
        self.trainer.datamodule.hparams.batch_size = self.num_samples_rt_60_tuning
        new_loader = self.trainer.datamodule.train_dataloader()
        batch = next(iter(new_loader))
        batch = self.trainer.datamodule.on_before_batch_transfer(batch, dataloader_idx=None)
        batch = tuple_to_device(batch, device=self.device)
        batch = self.trainer.datamodule.on_after_batch_transfer(batch, dataloader_idx=None)
        y, (s, h, rir_properties) = batch
        target_sigma, target_rt_60 = self.oracle_polack_analysis(h)
        self.rt_60_estimator.fit(y, target_rt_60, savefig_path=self.trainer.logger.log_dir)
        self.trainer.datamodule.hparams.batch_size = orig_batch_size
        potential_new_loader_id = id(self.trainer.train_dataloader)
        if orig_loader_id != potential_new_loader_id:
            raise RuntimeError("training dataloader has been replaced")

    # def validation_step(self):
    #     # ou sinon recalculer le DRR et le RT60 à partir de la rir synthéthisée
    #     ...

    def normalize_rir(self, h):
        normalization_method = self.synthesis_normalization_method
        if normalization_method is None or normalization_method == "" or "none" in normalization_method.lower():
            return h
        if "peak" in normalization_method.lower():
            return h / (h[..., 0].abs().unsqueeze(-1))
        if "total" in normalization_method.lower() and "energy" in normalization_method.lower():
            return h / h.abs().square().sum(dim=-1, keepdims=True).sqrt()
        if "rms" in normalization_method.lower():
            return h / h.abs().square().mean(dim=-1, keepdims=True).sqrt()
        else:
            raise ValueError()


class RT60EstimatorToPolack(AbsReverbModel):
    def __init__(
        self,
        rt_60_estimator: RT60Estimator,
        polack_synthesis: PolackSynthesis,
        metrics: list[nn.Module] = [],
    ):
        super().__init__(metrics=metrics, crop_input_to_target=False)
        warnings.warn("This method should no longer be used for training, please use BlindPolack instead")
        self.rt_60_estimator = rt_60_estimator
        self.polack_synthesis = polack_synthesis
        if self.polack_synthesis.fixed_sigma is None:
            raise NotImplementedError("needs to not depend on oracle rir properties")
        if not isinstance(self.polack_synthesis.early_echoes_masking_module, FixedTimeEarlyEnd):
            raise ValueError("Needs to not depend on rir properties")

    def forward(self, y):
        rt_60 = self.rt_60_estimator(y)[:, None, None]
        h_hat = self.polack_synthesis(sigma=torch.full_like(rt_60, torch.nan), rt_60=rt_60, rir_properties=dict())
        return h_hat

    def internal_loss(self, *args, **kwargs):
        return None

    # get_time and get_stft are as default


# %% Tests


def test_rt_60_divergence_with_oracle_properties(
    num_batches=10, regression_min_energy=-25, rt_60_depends_from_slope_only=True, normalization_method=None
):
    import lightning.pytorch as L

    L.seed_everything(12)
    from datasets import (
        WSJSimulatedRirDataModule,
        WSJ1SimulatedRirDataModule,
        SynthethicRirDataset,
        EARSReverbDataModule,
    )
    import matplotlib.pyplot as plt
    import scipy.stats
    import tqdm.auto as tqdm

    res_dict = dict()
    for data_module_name in ("Synthethic RIRs", "RIRs from EARS @48kHz", "RIRs from EARS @16kHz"):
        L.seed_everything(12)
        if "ears" in data_module_name.lower():
            if "48" in data_module_name.lower():
                data_module = EARSReverbDataModule(
                    batch_size=10, return_rir=True, resample_rir=False, enable_caching_val=False
                )
                polack_analysis_rt_60 = PolackAnalysisRT60Only(
                    rir_length=192000,
                    fs=48000,
                    regression_min_energy=regression_min_energy,
                    rt_60_depends_from_slope_only=rt_60_depends_from_slope_only,
                )
            else:
                data_module = EARSReverbDataModule(
                    batch_size=10, return_rir=True, resample_rir=True, enable_caching_val=False
                )
                polack_analysis_rt_60 = PolackAnalysisRT60Only(
                    rir_length=64000,
                    fs=16000,
                    regression_min_energy=regression_min_energy,
                    rt_60_depends_from_slope_only=rt_60_depends_from_slope_only,
                )
        else:
            data_module = WSJ1SimulatedRirDataModule(
                rir_dataset=SynthethicRirDataset(rir_root="./data/rirs_v2"),
                batch_size=10,
            )
            polack_analysis_rt_60 = PolackAnalysisRT60Only(
                rir_length=16383,
                fs=16000,
                regression_min_energy=regression_min_energy,
                rt_60_depends_from_slope_only=rt_60_depends_from_slope_only,
            )
        targets, preds = [], []
        data_module.prepare_data()
        data_module.setup()
        loader = data_module.train_dataloader()
        for i, (y, (x, h, rir_properties)) in enumerate(tqdm.tqdm(loader, total=min(len(loader), num_batches))):
            if i >= num_batches:
                break
            if normalization_method is None:
                h_normalized = h
            elif "peak" in normalization_method.lower():
                h_normalized = h / h[..., 0, None]
            elif "total" in normalization_method.lower() and "energy" in normalization_method.lower():
                h_normalized = h / h.abs().sqrt().sum().sqrt()
            elif "rms" in normalization_method.lower():
                h_normalized = h / h.abs().sqrt().mean().sqrt()
            else:
                raise ValueError()
            polack_analysis_rt_60.plot = i == 0
            preds.append(polack_analysis_rt_60(h_normalized))
            targets.append(rir_properties["rt_60"])
        preds = torch.cat(preds).squeeze().cpu()
        targets = torch.cat(targets).squeeze().cpu()
        print(data_module_name)
        print("MSE", nn.MSELoss()(preds, targets))
        print("MAE", nn.L1Loss()(preds, targets))
        print(
            "Pearson correlation",
            round(scipy.stats.pearsonr(preds.numpy(), targets.numpy()).statistic, 2),
        )
        res_dict[data_module_name] = (preds, targets)

    fig, axs = plt.subplots(1, len(res_dict), sharex=True, sharey=True, figsize=(30, 10))
    fig_suptitle = r"$RT_{60}$ distribution"
    fig.suptitle(fig_suptitle)
    upper_graph_lim = max(max((max(preds), max(targets)) for preds, targets in res_dict.values())) + 0.2
    lower_graph_lim = min(min((min(preds), min(targets)) for preds, targets in res_dict.values())) - 0.2
    for ax, (data_module_name, (preds, targets)) in zip(axs, res_dict.items()):
        ax.set_title(data_module_name)
        ax.set_aspect("equal")
        ax.plot(targets.sort().values, preds[targets.sort().indices], ".")
        ax.set_xlim(lower_graph_lim, upper_graph_lim)
        ax.set_ylim(lower_graph_lim, upper_graph_lim)
        ax.plot([lower_graph_lim, upper_graph_lim], [lower_graph_lim, upper_graph_lim], "--", color="tab:red")
        ax.set_xlabel("Oracle ${RT}_{60}$")
        ax.set_ylabel("Estimated ${RT}_{60}$")
    fig.savefig("res/2025_01_30/rt_60_distribution.pdf")
    fig.show()


def measure_energy_deviation_after_noise_floor(
    num_batches=20,
    regression_min_energy=-25,
    measure_energy_divergence_below=-35,
    normalize_energy: str = "peak energy",
    direct_path_duration_ms=0.0,
):
    import lightning.pytorch as L

    L.seed_everything(12)
    from datasets import (
        WSJSimulatedRirDataModule,
        WSJ1SimulatedRirDataModule,
        SynthethicRirDataset,
        EARSReverbDataModule,
    )
    import matplotlib.pyplot as plt
    import scipy.stats
    import tqdm.auto as tqdm

    from model.reverb_models.polack import PolackAnalysis
    import lightning.pytorch as L

    res_dict = dict()
    for data_module_name in ("Synthethic RIRs", "RIRs from EARS @48kHz", "RIRs from EARS @16kHz"):
        if "ears" in data_module_name.lower():
            if "48" in data_module_name.lower():
                data_module = EARSReverbDataModule(
                    batch_size=10, return_rir=True, resample_rir=False, enable_caching_val=False
                )
                polack_analysis = PolackAnalysis(
                    rir_length=192000,
                    fs=48000,
                    regression_min_energy=regression_min_energy,
                    direct_path_duration_ms=direct_path_duration_ms,
                )
            else:
                data_module = EARSReverbDataModule(
                    batch_size=10, return_rir=True, resample_rir=True, enable_caching_val=False
                )
                polack_analysis = PolackAnalysis(
                    rir_length=64000,
                    fs=16000,
                    regression_min_energy=regression_min_energy,
                    direct_path_duration_ms=direct_path_duration_ms,
                )
        else:
            data_module = WSJ1SimulatedRirDataModule(
                rir_dataset=SynthethicRirDataset(rir_root="./data/rirs_v2"),
                batch_size=10,
            )
            polack_analysis = PolackAnalysis(
                rir_length=16383,
                fs=16000,
                regression_min_energy=regression_min_energy,
                direct_path_duration_ms=direct_path_duration_ms,
            )

        L.seed_everything(12)
        polack_energies, observed_energies = [], []
        data_module.prepare_data()
        data_module.setup()
        loader = data_module.train_dataloader()
        for i, (y, (x, h, rir_properties)) in enumerate(tqdm.tqdm(loader, total=min(len(loader), num_batches))):
            if i >= num_batches:
                break
            polack_analysis.plot = i == 0
            # RMS normalization messes up everything, so we only divide h by its direct path energy
            # h = h / (h.abs().square().sum(axis=-1, keepdim=True).sqrt())
            if "peak" in normalize_energy.lower() and "energy" in normalize_energy.lower():
                h = h / (
                    h[..., 0 : round(polack_analysis.fs * 0.0025)].abs().square().sum(axis=-1, keepdim=True).sqrt()
                )
            elif "peak" in normalize_energy:
                h = h / h.abs()[..., 0, None]
            else:
                pass
            sigma, rt_60 = polack_analysis(h)
            edc_h = edc(h)  # + self.epsilon
            edc_db = energy_to_db(edc_h)
            total_energy = edc_db[..., 0, None]
            edc_db_scaled = edc_db - total_energy

            begin_measured_divergence_in_samples = torch.argmin(
                1 * (edc_db_scaled > measure_energy_divergence_below), dim=-1, keepdim=True
            )

            # end_polack_in_samples = begin_polack_in_samples
            # observed_energy = edc_h.gather(dim=-1, index=end_polack_in_samples)
            polack_energy_ = polack_energy(
                sigma, rt_60, integral_lower_bound=begin_measured_divergence_in_samples, fs=polack_analysis.fs
            )
            h_cloned = h.clone()
            h_cloned[edc_db_scaled > measure_energy_divergence_below] = torch.nan
            observed_energy = h_cloned.abs().square().nansum(-1, keepdim=True)
            observed_energies.append(observed_energy)
            polack_energies.append(polack_energy_)

        observed_energies = torch.cat(observed_energies).squeeze().cpu()
        polack_energies = torch.cat(polack_energies).squeeze().cpu()
        assert (observed_energies > 0).all()
        assert (polack_energies > 0).all()
        print("MSE", nn.MSELoss()(observed_energies, polack_energies))
        print("MAE", nn.L1Loss()(observed_energies, polack_energies))
        print(
            "Pearson correlation",
            round(scipy.stats.pearsonr(observed_energies.numpy(), polack_energies.numpy()).statistic, 2),
        )
        res_dict[data_module_name] = (polack_energies, observed_energies)

    fig, axs = plt.subplots(1, len(res_dict), sharex=True, sharey=True, figsize=(30, 10))
    fig_suptitle = f"Energy distribution below {round(measure_energy_divergence_below)} dB EDC"
    fig.suptitle(fig_suptitle)
    upper_graph_lim = (
        max(
            max(
                (max(observed_energies), max(polack_energies))
                for polack_energies, observed_energies in res_dict.values()
            )
        )
        + 0.001
    )
    lower_graph_lim = (
        min(
            min(
                (min(observed_energies), min(polack_energies))
                for polack_energies, observed_energies in res_dict.values()
            )
        )
        - 0.001
    )
    for ax, (data_module_name, (polack_energies, observed_energies)) in zip(axs, res_dict.items()):
        ax.set_title(data_module_name)
        ax.set_aspect("equal")
        ax.plot(polack_energies, observed_energies, ".")
        ax.set_xlim(lower_graph_lim, upper_graph_lim)
        ax.set_ylim(lower_graph_lim, upper_graph_lim)
        ax.plot([lower_graph_lim, upper_graph_lim], [lower_graph_lim, upper_graph_lim], "--", color="tab:red")
        ax.set_xlabel(f"Polack energy")
        ax.set_ylabel("Observed energy")
    fig.savefig("res/2025_01_30/" + fig_suptitle + ".pdf")
    fig.show()


def test_polack(analysis_normalization_method=None, move_all_direct_energy_to_peak=False):
    from datasets import (
        WSJSimulatedRirDataModule,
        SynthethicRirDataset,
        WSJDataset,
        EarsOnlyRIRsDataset,
    )
    import matplotlib.pyplot as plt
    from model.reverb_models.early_echoes import MeanFreePathEarlyEnd, FixedTimeEarlyEnd

    # plt.close("all")

    # torch.manual_seed(0)
    rir_dataset = SynthethicRirDataset(query="rt_60>0.8")
    # rir_dataset = SynthethicRirDataset(query="rt_60<0.3")
    WSJDataset("data/speech", "test")
    # rir_dataset = SynthethicRirDataset(query='rt_60 > 0.5 and rt_60 < 0.6')
    # rir_dataset = SynthethicRirDataset(query="rt_60 > 0. and rt_60 < 0.3")
    data_module = WSJSimulatedRirDataModule(rir_dataset=rir_dataset, dry_signal_target_len=49151, batch_size=2)
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()

    # h = h[0]

    polack_analysis_synthesis = OraclePolackAnalysisSynthesis(
        early_echoes_masking_module=FixedTimeEarlyEnd(end_time=0.0),
        regress=True,
        regression_max_energy=-5.0,
        regression_min_energy=-25.0,
        intersect_zero=False,
        rt_60_depends_from_slope_only=True,
        analysis_normalization_method=analysis_normalization_method,
        analysis_rir_length=16383,
        synthesis_rir_length=16383,
        analysis_fs=16000,
        synthesis_fs=16000,
        positive_valued=True,
        num_polack_draws=1,
        move_all_direct_energy_to_peak=move_all_direct_energy_to_peak,
    )
    polack_analysis_synthesis.oracle_polack_analysis.plot = True

    print("Analysis from knwon parameters")
    rt_60 = 0.8
    sigma = 0.1 * torch.ones(2, 1, 1)
    h = polack_analysis_synthesis.polack_synthesis(sigma, rt_60, {})
    h[..., 0] = 1
    print("normalization from peak")

    sigma_hat, rt_60_hat = polack_analysis_synthesis.oracle_polack_analysis(h)
    print(f"Sigma: estimated={sigma_hat}, ground truth: {sigma}")
    print(f"RT60: estimated={rt_60_hat}, ground truth: {rt_60}")

    print("from ISM RIR")
    y, (x, h, rir_properties) = next(iter(loader))
    h_hat = polack_analysis_synthesis.training_step((y, (h, rir_properties)), batch_idx=0)["pred"]
    sigma_hat, rt_60_hat = polack_analysis_synthesis.oracle_polack_analysis(h)
    print(f"RT60: estimated={rt_60_hat}, ground truth: {rir_properties['rt_60']}")
    fig_ext, ax_ext = plt.subplots()
    fig_ext.suptitle("ISM")
    ax_ext.plot(polack_analysis_synthesis.polack_synthesis.T, h_hat[0].squeeze(), label="synthesized")
    ax_ext.plot(polack_analysis_synthesis.polack_synthesis.T, h[0].squeeze(), label="tgt")
    ax_ext.legend()
    fig_ext.show()

    print("from EARS RIR")
    dataset_name = "EARS"
    import torchaudio
    import glob
    import os
    from model.utils.tensor_ops import crop_or_zero_pad_to_target_len

    rir_dataset = EarsOnlyRIRsDataset(rir_root="./data/speech/EARS-Reverb/train_unique_rirs/", fs=16000)
    h, rir_properties = rir_dataset[12]
    h = crop_or_zero_pad_to_target_len(h[None, ...], 16383)
    polack_analysis_synthesis.polack_synthesis.positive_valued = False
    h_hat = polack_analysis_synthesis.training_step((y, (h, rir_properties)), batch_idx=0)["pred"]
    fig_ext, ax_ext = plt.subplots()
    fig_ext.suptitle("EARS RIR")
    ax_ext.plot(polack_analysis_synthesis.polack_synthesis.T, h_hat[0].squeeze(), label="synthesized")
    ax_ext.plot(polack_analysis_synthesis.polack_synthesis.T, h[0].squeeze(), label="tgt")
    ax_ext.legend()
    fig_ext.show()

    print("With resampling")
    polack_analysis_synthesis = OraclePolackAnalysisSynthesis(
        early_echoes_masking_module=FixedTimeEarlyEnd(end_time=0.0, fs=16000, rir_length=64000),
        regress=True,
        regression_max_energy=-5.0,
        regression_min_energy=-25.0,
        intersect_zero=False,
        rt_60_depends_from_slope_only=True,
        analysis_normalization_method=analysis_normalization_method,
        analysis_rir_length=192000,
        synthesis_rir_length=64000,
        analysis_fs=48000,
        synthesis_fs=16000,
        positive_valued=False,
        num_polack_draws=1,
        move_all_direct_energy_to_peak=move_all_direct_energy_to_peak,
    )
    polack_analysis_synthesis.oracle_polack_analysis.plot = True

    rir_dataset = EarsOnlyRIRsDataset(rir_root="./data/speech/EARS-Reverb/train_unique_rirs/", fs=48000)
    # dataset_idx=12 # marche pas bien car oscillations
    dataset_idx = 0
    h, rir_properties = rir_dataset[dataset_idx]
    h = crop_or_zero_pad_to_target_len(h[None, ...], 192000)
    h_hat = polack_analysis_synthesis.training_step((y, (h, rir_properties)), batch_idx=0)["pred"]
    fig_ext, ax_ext = plt.subplots()
    fig_ext.suptitle("EARS RIR (48kHz)")
    ax_ext.plot(polack_analysis_synthesis.polack_synthesis.T, h_hat[0].squeeze(), label="synthesized")
    ax_ext.plot(
        polack_analysis_synthesis.oracle_polack_analysis.orig_indexes_for_regression, h[0].squeeze(), label="tgt"
    )
    ax_ext.legend()
    fig_ext.show()


def position_dry_synth_wet_on_plane(*, loss_dry_wet, loss_synthesized_dry, loss_synthesized_wet, center="wet"):
    dry = (0.0, 0.0)
    wet = (math.sqrt(loss_dry_wet), 0.0)
    synthesized_x = (loss_synthesized_dry - loss_synthesized_wet + loss_dry_wet) / (2 * (wet[0]))
    synthesized_y = math.sqrt(loss_synthesized_dry - synthesized_x**2)
    synthesized = (synthesized_x, synthesized_y)
    if "wet" in center.lower():
        diff = wet[0] - dry[0]
        dry = (dry[0] - diff, dry[1])
        synthesized = (synthesized[0] - diff, synthesized[1])
        wet = (wet[0] - diff, wet[1])
    return dry, synthesized, wet


def test_polack_analysis_synthesis(
    num_batches=1,
    normalize_by_loss_dry_wet=True,
    reverberation_time_factor=1 / 2,
    drr_difference_db=3.0,
    device="cuda",
    plot_examples=True,
    num_optim_steps=1,
):
    # Test all normalization methods and parameters for direct-path energy distribution
    # Use plot
    import lightning.pytorch as L
    import itertools
    from datasets import EARSReverbDataModule, WSJ1SimulatedRirDataModule, SynthethicRirDataset
    from model.utils.tensor_ops import fftconvolve, db_to_amplitude, tuple_to_device
    from model.reverb_models.early_echoes import MeanFreePathEarlyEnd, FixedTimeEarlyEnd
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    from model.losses.rereverberation_loss import TimeFrequencyRereverberationLoss
    from model.losses.base_losses import SumLosses, ComplexToRealMSELoss, LogLoss, ScaleInvariant
    import numpy as np
    import scipy.stats
    import os
    import datetime

    os.makedirs("res/" + str(datetime.date.today()), exist_ok=True)
    if "cuda" in device.lower() and torch.cuda.mem_get_info()[1] < 26 * (1024**3):
        # if there is less than 26 Gb GPU VRAM
        num_absolute_cross_bands = 2
        synthesis_rir_length_for_ears = 16000
        num_polack_draws_if_several = 5
    else:
        num_absolute_cross_bands = 4
        synthesis_rir_length_for_ears = 32000
        num_polack_draws_if_several = 10

    tf_loss_module = TimeFrequencyRereverberationLoss(
        base_loss=ScaleInvariant(SumLosses(ComplexToRealMSELoss(), LogLoss(epsilon=1.0))),
        num_absolute_cross_bands=num_absolute_cross_bands,
    ).to(device=device)
    # tf_loss_module = TimeFrequencyRereverberationLoss(
    #     base_loss=ScaleInvariant(ComplexToRealMSELoss()), num_absolute_cross_bands=4
    # ).to(device=device)

    reverberation_shortening = ReverberationTimeShortening().to(device=device)

    plt.close("all")
    batch_size = 1
    res = dict()
    for data_module_name in (
        "Synthethic RIRs",
        "RIRs from EARS @48kHz",
        "RIRs from EARS @16kHz",
    ):
        if "ears" in data_module_name.lower():
            if "48" in data_module_name.lower():
                data_module = EARSReverbDataModule(
                    batch_size=batch_size, return_rir=True, resample_rir=False, enable_caching_val=False
                )
                analysis_fs = 48000
                analysis_rir_length = 192000
                synthesis_rir_length = synthesis_rir_length_for_ears
            else:
                data_module = EARSReverbDataModule(
                    batch_size=batch_size, return_rir=True, resample_rir=True, enable_caching_val=False
                )
                analysis_fs = 16000
                analysis_rir_length = 64000
                synthesis_rir_length = synthesis_rir_length_for_ears
        else:
            data_module = WSJ1SimulatedRirDataModule(
                rir_dataset=SynthethicRirDataset(rir_root="./data/rirs_v2"),
                batch_size=batch_size,
            )
            analysis_fs = 16000
            analysis_rir_length = 16383
            synthesis_rir_length = 16383
        res[data_module_name] = dict()
        L.seed_everything(12)
        data_module.prepare_data()
        data_module.setup()
        rir_resampling_transfom = (
            data_module.dataset_train.resampling_transform
            if "ears" in data_module_name.lower() and "48" in data_module_name.lower()
            else torch.nn.Identity()
        )
        print(10 * "-", data_module_name, 10 * "-")
        plotted_filters = set()
        for (
            analysis_normalization_method,
            synthesis_normalization_method,
            move_all_direct_energy_to_peak,
            early_echoes_follow_polack,
            positive_valued,
            num_evenly_spaced_synthesis_filters,
            num_polack_draws,
        ) in itertools.product(
            (
                # "peak",
                # "direct energy",
                # "total energy",
                "None",
            ),
            (
                # "peak",
                # "direct energy",
                # "total energy",
                "None",
            ),
            (
                True,
                # False,
            ),
            (
                True,
                # False,
            ),
            (
                True,
                False,
            ),
            (
                1,
                30,
                257,
                # 50,
            ),
            (
                1,
                num_polack_draws_if_several,
                # 10,
            ),
        ):
            if early_echoes_follow_polack:
                early_echoes_end_time = 0.0025
            else:
                early_echoes_end_time = 0.050
            polack_kwargs = dict(
                early_echoes_masking_module=FixedTimeEarlyEnd(
                    end_time=early_echoes_end_time, fs=16000, rir_length=synthesis_rir_length
                ),
                regress=True,
                regression_max_energy=-5.0,
                regression_min_energy=-25.0,
                intersect_zero=False,
                rt_60_depends_from_slope_only=True,
                analysis_normalization_method=analysis_normalization_method,
                synthesis_normalization_method=synthesis_normalization_method,
                analysis_rir_length=analysis_rir_length,
                synthesis_rir_length=synthesis_rir_length,
                analysis_fs=analysis_fs,
                synthesis_fs=16000,
                positive_valued=positive_valued,
                num_polack_draws=num_polack_draws,
                move_all_direct_energy_to_peak=move_all_direct_energy_to_peak,
                direct_path_duration_ms=2.5,
            )

            if 1 < num_evenly_spaced_synthesis_filters <= 32:
                # polack_kwargs["regression_min_energy"] = -20.0
                polack_analysis_synthesis = OracleBandwisePolackAnalysisSynthesis(
                    num_evenly_spaced_synthesis_filters=num_evenly_spaced_synthesis_filters, **polack_kwargs
                )
                if num_evenly_spaced_synthesis_filters not in plotted_filters:
                    plotted_filters.add(num_evenly_spaced_synthesis_filters)
                    polack_analysis_synthesis.analysis_filterbank.plot_filters()
                    polack_analysis_synthesis.synthesis_filterbank.plot_filters()
            elif 32 < num_evenly_spaced_synthesis_filters:
                synthesis_n_fft = (num_evenly_spaced_synthesis_filters - 1) * 2
                polack_analysis_synthesis = OracleSTFTPolackAnalysisSynthesis(
                    synthesis_n_fft=synthesis_n_fft,
                    use_GL=True,
                    **polack_kwargs,
                )
            else:
                polack_analysis_synthesis = OraclePolackAnalysisSynthesis(**polack_kwargs)
                # Set analysis that will be kept for STFTPolack
                time_domain_polack_analysis = polack_analysis_synthesis.oracle_polack_analysis
                drr_analysis = polack_analysis_synthesis.oracle_drr_module_for_analysis
                drr_analysis.polack_analysis = polack_analysis_synthesis.oracle_polack_analysis

            polack_analysis_synthesis.plot = False
            polack_analysis_synthesis = polack_analysis_synthesis.to(device=device)
            second_polack_analysis = PolackAnalysis(
                regress=True,
                regression_max_energy=-5,
                regression_min_energy=-25,
                rir_length=synthesis_rir_length,
                fs=16000,
                intersect_zero=False,
                rt_60_depends_from_slope_only=True,
                direct_path_duration_ms=2.5,
            ).to(device=device)

            drr_synthesis = DRR(fs=16000).to(device=device)

            L.seed_everything(12)
            loader = data_module.train_dataloader()
            parameters_str = (
                f"analysis normalization={analysis_normalization_method},\n"
                + f"synthesis_normalization={synthesis_normalization_method},\n"
                + f"move_direct_energy_to_peak={move_all_direct_energy_to_peak},\n"
                + f"early_echoes_follow_polack={early_echoes_follow_polack},\n"
                + f"positive_polack_model={positive_valued}\n"
                + f"num_synthesis_filters={num_evenly_spaced_synthesis_filters}\n"
                + f"num_polack_draws={num_polack_draws}\n"
            )
            print(parameters_str)
            res[data_module_name][parameters_str] = dict(
                tf_loss_difference=np.full(num_batches, np.nan),
                SISDR_difference=np.full(num_batches, np.nan),
                correlation=0.0,
                loss_dry_wet=np.full(num_batches, np.nan),
                loss_synthesized_wet=np.full(num_batches, np.nan),
                loss_synthesized_dry=np.full(num_batches, np.nan),
                loss_S_hat_dry=np.full(num_batches, np.nan),
                loss_S_hat_wet=np.full(num_batches, np.nan),
                loss_shortened_wet=np.full(num_batches, np.nan),
                loss_shortened_dry=np.full(num_batches, np.nan),
                loss_drrmodified_wet=np.full(num_batches, np.nan),
                loss_drrmodified_dry=np.full(num_batches, np.nan),
                loss_Y_hat_wet=np.full(num_batches, np.nan),
                loss_Y_hat_dry=np.full(num_batches, np.nan),
                position_dry=np.full((num_batches, 2), np.nan),
                position_wet=np.full((num_batches, 2), np.nan),
                position_synthesized=np.full((num_batches, 2), np.nan),
                position_S_hat=np.full((num_batches, 2), np.nan),
                position_shortened=np.full((num_batches, 2), np.nan),
                position_drrmodified=np.full((num_batches, 2), np.nan),
                position_Y_hat=np.full((num_batches, 2), np.nan),
                original_drr=np.full(num_batches, np.nan),
            )
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                (y, (s, h, rir_properties)) = tuple_to_device(batch, device=device)
                polack_analysis_synthesis.plot = i == 0 and plot_examples
                polack_analysis_synthesis.oracle_polack_analysis.plot = i == 0 and plot_examples

                h_hat = polack_analysis_synthesis(h.clone(), rir_properties)

                orig_sigma, orig_rt_60 = time_domain_polack_analysis(h)
                synthesized_sigma, synthesized_rt_60 = second_polack_analysis(h_hat)
                print(f"sigma: {orig_sigma.squeeze().item():.2e} -> {synthesized_sigma[0].squeeze().mean().item():.2e}")
                print(f"rt_60: {orig_rt_60.squeeze().item():.2f} -> {synthesized_rt_60[0].squeeze().mean().item():.2f}")

                # h_hat = torch.sign(h[..., 0, None]) * h_hat

                if "ears" in data_module_name.lower() and i == 0:
                    if getattr(polack_analysis_synthesis.oracle_polack_analysis, "plot", False):
                        plt.title(parameters_str)
                    h_16k = rir_resampling_transfom(h.cpu()).to(device=device)
                else:
                    h_16k = h
                # if i == 0:
                #     plt.figure()
                #     plt.plot(y[0].squeeze() / y[0].squeeze().abs().square().mean().sqrt(), label="wet")
                #     plt.plot(s[0].squeeze() / s[0].squeeze().abs().square().mean().sqrt(), label="dry")
                #     plt.plot(y_hat[0].squeeze() / y_hat[0].squeeze().abs().square().mean().sqrt(), label="synthesized")
                #     plt.legend()
                #     plt.show()
                h_16k_normalized = polack_analysis_synthesis.normalize_rir(h_16k, stage="synthesis")[
                    ..., :synthesis_rir_length
                ]
                print()
                original_drr = drr_analysis(h).item()
                res[data_module_name][parameters_str]["original_drr"][i] = original_drr
                print(f"Polack DRR: {original_drr:.2f} -> {drr_synthesis(h_hat)[0].squeeze().mean().item():.2f}")
                print(
                    f"Direct energy: {drr_analysis.direct_energy(h)[0].squeeze().mean().item():.2e}"
                    + f" ->  {drr_synthesis.direct_energy(h_hat)[0].squeeze().mean().item():.2e}"
                )
                print(
                    f"Reverberant energy: {drr_analysis.reverberant_energy(h)[0].squeeze().mean().item():.2e}"
                    + f" ->  {drr_synthesis.reverberant_energy(h_hat)[0].squeeze().mean().item():.2e}"
                )
                h_shortened = reverberation_shortening(
                    h_16k_normalized, rir_properties["rt_60"], rir_properties["rt_60"] * reverberation_time_factor
                )
                h_drrmodified = h_16k_normalized.clone()
                orig_direct_energy_db = energy_to_db(
                    polack_analysis_synthesis.oracle_drr_module_for_analysis.direct_energy(h)
                )
                # new_direct_energy = orig_direct_energy_db + drr_difference_db
                gain_linear = db_to_amplitude(drr_difference_db * torch.ones_like(h[..., 0]))
                h_drrmodified[..., : round(16000 * 0.0025)] = (
                    gain_linear * h_16k_normalized[..., : round(16000 * 0.0025)]
                )
                print(
                    f"Classical DRR (modified by {drr_difference_db:+.1f} dB): {drr_synthesis(h_16k_normalized).item():.2f} -> {drr_synthesis(h_drrmodified).item():.2f}"
                )

                y = fftconvolve(s, h_16k_normalized)[..., : s.size(-1)]
                # sisdr_dry_wet = ScaleInvariantSignalDistortionRatio()(s[..., : y.size(-1)], y[..., : s.size(-1)]).item()
                # sisdr_synthesized_wet = ScaleInvariantSignalDistortionRatio()(
                # y_hat[..., : s.size(-1)], y[..., : s.size(-1)]
                # ).item()
                # print(f"SISDR between synthesized and reverberant ({parameters_str})", sisdr_synthesized_wet)
                # print(f"SISDR between dry and reverberant", sisdr_dry_wet)
                # print(
                #     f"SISDR improvement from dry to synthesized \n {parameters_str}: \n {(sisdr_synthesized_wet - sisdr_dry_wet):.2f}"
                # )
                S = tf_loss_module.stft_module(s)
                Y = tf_loss_module.stft_module(y)
                dirac = torch.zeros_like(h_16k_normalized)
                dirac[..., 0] = 1.0
                dirac_normalized = polack_analysis_synthesis.normalize_rir(dirac, stage="synthesis")
                S_hat = S.clone()
                Y_hat = Y.clone()
                S_hat.requires_grad = True
                Y_hat.requires_grad = True
                optim_S_hat = torch.optim.Adam(params=(S_hat,))
                optim_Y_hat = torch.optim.Adam(params=(Y_hat,))
                for _ in range(num_optim_steps):
                    h_hat = polack_analysis_synthesis(h.clone(), rir_properties)
                    loss_S_hat_wet = tf_loss_module(S_hat, h_hat, None, None, y)
                    loss_Y_hat_wet = tf_loss_module(Y_hat, h_hat, None, None, y)
                    with torch.no_grad():
                        loss_S_hat_wet.backward()
                        loss_Y_hat_wet.backward()
                        optim_S_hat.step()
                        optim_Y_hat.step()
                        optim_S_hat.zero_grad()
                        optim_Y_hat.zero_grad()
                with torch.no_grad():
                    loss_synthesized_wet = tf_loss_module(S, h_hat, None, None, y).item()
                    loss_dry_wet = tf_loss_module(S, dirac_normalized, None, None, y).item()
                    loss_S_hat_wet = tf_loss_module(S_hat, dirac_normalized, None, None, y).item()
                    loss_S_hat_dry = tf_loss_module(S_hat, dirac_normalized, None, None, s).item()
                    loss_synthesized_dry = tf_loss_module(S, h_hat, None, None, s).item()
                    loss_shortened_dry = tf_loss_module(S, h_shortened, None, None, s).item()
                    loss_shortened_wet = tf_loss_module(S, h_shortened, None, None, y).item()
                    loss_drrmodified_dry = tf_loss_module(S, h_drrmodified, None, None, s).item()
                    loss_drrmodified_wet = tf_loss_module(S, h_drrmodified, None, None, y).item()
                    loss_Y_hat_wet = tf_loss_module(Y_hat, dirac_normalized, None, None, y).item()
                    loss_Y_hat_dry = tf_loss_module(Y_hat, dirac_normalized, None, None, s).item()

                    # print(f"Model loss difference: {():.2f}")
                    # res[data_module_name][parameters_str]["tf_loss_difference"][i] = loss_synthesized_wet - loss_dry_wet
                    # res[data_module_name][parameters_str]["SISDR_difference"][i] = sisdr_synthesized_wet - sisdr_dry_wet
                    if normalize_by_loss_dry_wet:
                        loss_synthesized_dry /= loss_dry_wet
                        loss_synthesized_wet /= loss_dry_wet
                        loss_S_hat_dry /= loss_dry_wet
                        loss_S_hat_wet /= loss_dry_wet
                        loss_shortened_wet /= loss_dry_wet
                        loss_shortened_dry /= loss_dry_wet
                        loss_drrmodified_dry /= loss_dry_wet
                        loss_drrmodified_wet /= loss_dry_wet
                        loss_Y_hat_wet /= loss_dry_wet
                        loss_Y_hat_dry /= loss_dry_wet
                        loss_dry_wet /= loss_dry_wet
                    res[data_module_name][parameters_str]["loss_dry_wet"][i] = loss_dry_wet
                    res[data_module_name][parameters_str]["loss_synthesized_wet"][i] = loss_synthesized_wet
                    res[data_module_name][parameters_str]["loss_synthesized_dry"][i] = loss_synthesized_dry
                    res[data_module_name][parameters_str]["loss_S_hat_wet"][i] = loss_S_hat_wet
                    res[data_module_name][parameters_str]["loss_S_hat_dry"][i] = loss_S_hat_dry
                    res[data_module_name][parameters_str]["loss_shortened_dry"][i] = loss_shortened_dry
                    res[data_module_name][parameters_str]["loss_shortened_wet"][i] = loss_shortened_wet
                    res[data_module_name][parameters_str]["loss_drrmodified_dry"][i] = loss_drrmodified_dry
                    res[data_module_name][parameters_str]["loss_drrmodified_wet"][i] = loss_drrmodified_wet
                    res[data_module_name][parameters_str]["loss_Y_hat_dry"][i] = loss_Y_hat_dry
                    res[data_module_name][parameters_str]["loss_Y_hat_wet"][i] = loss_Y_hat_wet

                    position_dry, position_synthesized, position_wet = position_dry_synth_wet_on_plane(
                        loss_dry_wet=loss_dry_wet,
                        loss_synthesized_dry=loss_synthesized_dry,
                        loss_synthesized_wet=loss_synthesized_wet,
                    )
                    _, position_S_hat, _ = position_dry_synth_wet_on_plane(
                        loss_dry_wet=loss_dry_wet,
                        loss_synthesized_dry=loss_S_hat_dry,
                        loss_synthesized_wet=loss_S_hat_wet,
                    )
                    _, position_shortened, _ = position_dry_synth_wet_on_plane(
                        loss_dry_wet=loss_dry_wet,
                        loss_synthesized_dry=loss_shortened_dry,
                        loss_synthesized_wet=loss_shortened_wet,
                    )

                    _, position_drrmodified, _ = position_dry_synth_wet_on_plane(
                        loss_dry_wet=loss_dry_wet,
                        loss_synthesized_dry=loss_drrmodified_dry,
                        loss_synthesized_wet=loss_drrmodified_wet,
                    )

                    _, position_Y_hat, _ = position_dry_synth_wet_on_plane(
                        loss_dry_wet=loss_dry_wet,
                        loss_synthesized_dry=loss_Y_hat_dry,
                        loss_synthesized_wet=loss_Y_hat_wet,
                    )
                    res[data_module_name][parameters_str]["position_dry"][i] = position_dry
                    res[data_module_name][parameters_str]["position_synthesized"][i] = position_synthesized
                    res[data_module_name][parameters_str]["position_S_hat"][i] = position_S_hat
                    res[data_module_name][parameters_str]["position_wet"][i] = position_wet
                    res[data_module_name][parameters_str]["position_shortened"][i] = position_shortened
                    res[data_module_name][parameters_str]["position_drrmodified"][i] = position_drrmodified
                    res[data_module_name][parameters_str]["position_Y_hat"][i] = position_Y_hat

            # res[data_module_name][parameters_str]["correlation"] = scipy.stats.pearsonr(
            # res[data_module_name][parameters_str]["tf_loss_difference"],
            # res[data_module_name][parameters_str]["SISDR_difference"],
            # ).statistic
            fig, ax = plt.subplots(figsize=(10, 10))
            fig_title = data_module_name + "\n" + parameters_str
            ax.scatter(
                res[data_module_name][parameters_str]["position_dry"][:, 0],
                res[data_module_name][parameters_str]["position_dry"][:, 1],
                label="dry",
            )
            ax.scatter(
                res[data_module_name][parameters_str]["position_synthesized"][:, 0],
                res[data_module_name][parameters_str]["position_synthesized"][:, 1],
                label="synthesized",
            )
            ax.scatter(
                res[data_module_name][parameters_str]["position_wet"][:, 0],
                res[data_module_name][parameters_str]["position_wet"][:, 1],
                s=res[data_module_name][parameters_str]["original_drr"]
                - res[data_module_name][parameters_str]["original_drr"].min()
                + plt.rcParams["lines.markersize"] ** 2,
                label="wet",
            )
            ax.scatter(
                res[data_module_name][parameters_str]["position_S_hat"][:, 0],
                res[data_module_name][parameters_str]["position_S_hat"][:, 1],
                label=r"Adam optimizer on $\mathcal{L}(X, \hat{h}, y)$ with $X$ starting at $S$",
            )
            ax.scatter(
                res[data_module_name][parameters_str]["position_shortened"][:, 0],
                res[data_module_name][parameters_str]["position_shortened"][:, 1],
                label=f"RIR shortening (factor={reverberation_time_factor:.2f})",
            )
            ax.scatter(
                res[data_module_name][parameters_str]["position_drrmodified"][:, 0],
                res[data_module_name][parameters_str]["position_drrmodified"][:, 1],
                label=f"DRR modification ({drr_difference_db:+.1f} dB)",
            )
            ax.scatter(
                res[data_module_name][parameters_str]["position_Y_hat"][:, 0],
                res[data_module_name][parameters_str]["position_Y_hat"][:, 1],
                label=r"Adam optimizer on $\mathcal{L}(X, \hat{h}, y)$ with $X$ starting at $Y$",
                # label=r"$Y - $" + f" {gradient_step:.1f}" + r" $ \nabla_Y \mathcal{L}$",
            )

            ax.set_title(fig_title)
            ax.set_xlim(-1.25, 0.25)
            ax.set_ylim(-0.1, 1.4)
            ax.set_aspect("equal")
            ax.legend()
            fig.tight_layout()
            fig.savefig("res/" + str(datetime.date.today()) + "/" + fig_title.replace("\n", "-") + ".pdf")
            # fig.show()
            # print(f"L(dry, wet) = {res[data_module_name][parameters_str]['loss_dry_wet'].mean():.2e}")
            # print(f"L(synth, wet) = {res[data_module_name][parameters_str]['loss_synthesized_wet'].mean():.2e}")
            # print(f"L(synth, dry) = {res[data_module_name][parameters_str]['loss_synthesized_dry'].mean():.2e}")
            # print(f"tf loss difference {res[data_module_name][parameters_str]['tf_loss_difference'].mean():.2e}")
            # print(f"SISDR difference {res[data_module_name][parameters_str]['SISDR_difference'].mean():.2f}")
            # print(
            #     f"Pearson correlation between SISDR and Loss {res[data_module_name][parameters_str]['correlation']:.2f}"
            # )

        print()
        # dict_correlation = {k: res[data_module_name][k]["correlation"] for k in res[data_module_name].keys()}
        # print(f"best method for correlation: {min(dict_correlation, key=dict_correlation.get)}")
        # dict_sisdr = {k: res[data_module_name][k]["SISDR_difference"] for k in res[data_module_name].keys()}
        # print(f"best method for SISDR {max(dict_sisdr, key=lambda k: dict_sisdr.get(k).mean())}")
        # dict_loss = {k: res[data_module_name][k]["tf_loss_difference"] for k in res[data_module_name].keys()}
        # print(f"best method for Loss {min(dict_loss, key=lambda k: dict_loss.get(k).mean())}")
    return res


def test_stft_polack(num_batches=1, resample_rir=True, use_GL: bool = False):
    from datasets import EARSReverbDataModule
    from lightning.pytorch import seed_everything

    seed_everything(12)

    data_module = EARSReverbDataModule(
        batch_size=8,
        return_rir=True,
        resample_rir=resample_rir,
        enable_caching_val=False,
    )

    data_module.prepare_data()
    data_module.setup()
    seed_everything(12)
    loader = data_module.train_dataloader()
    early_echoes_masking_module = FixedTimeEarlyEnd(end_time=0.0025, fs=16000, rir_length=32000)
    module = STFTRIRToPolack(
        early_echoes_masking_module=early_echoes_masking_module,
        synthesis_n_fft=512,
        use_GL=use_GL,
        analysis_fs=16000 if resample_rir else 48000,
        analysis_rir_length=64000 if resample_rir else 192000,
        synthesis_rir_length=32000,
        analysis_normalization_method=None,
        synthesis_normalization_method=None,
    )
    module.plot = True
    module.oracle_polack_analysis.plot = True
    # module.polack_synthesis.plot = True
    for i, (y, (s, h, rir_properties)) in enumerate(loader):
        if i >= num_batches:
            break
        h_hat = module(h, dict())
        print(f"total energy: {h.norm().square().cpu().item()} -> {h_hat.norm().square().cpu().item()}")


def test_bandwise_polack(num_evenly_spaced_filters=4):
    raise DeprecationWarning("moved to test_polack_analysis_synthesis")
    raise NotImplementedError("TODO test DRR")
    from datasets import WSJSimulatedRirDataModule, SynthethicRirDataset, WSJDataset, AdaspRirDataset
    import matplotlib.pyplot as plt
    from model.reverb_models.early_echoes import MeanFreePathEarlyEnd

    early_echoes_masking_module = FixedTimeEarlyEnd()
    bandwise_polack_analysis_synthesis = OracleBandwisePolackAnalysisSynthesis(
        early_echoes_masking_module=early_echoes_masking_module,
        num_evenly_spaced_filters=num_evenly_spaced_filters,
        filter_order=8,
        regress=True,
        regression_max_energy=-3.0,
        regression_min_energy=-20.0,
        rir_length=16383,
        positive_valued=False,
        num_polack_draws=1,
    )
    polack_analysis_synthesis = OraclePolackAnalysisSynthesis(
        early_echoes_masking_module=early_echoes_masking_module,
        regress=True,
        regression_max_energy=-3.0,
        regression_min_energy=-20.0,
        rir_length=16383,
        positive_valued=False,
        num_polack_draws=1,
    )

    # torch.manual_seed(0)
    # rir_dataset = SynthethicRirDataset(query="rt_60>0.8")
    rir_dataset = AdaspRirDataset(["Arni"])
    # rir_dataset = SynthethicRirDataset(query="rt_60<0.3")
    WSJDataset("data/speech", "test")
    # rir_dataset = SynthethicRirDataset(query='rt_60 > 0.5 and rt_60 < 0.6')
    # rir_dataset = SynthethicRirDataset(query="rt_60 > 0. and rt_60 < 0.3")
    data_module = WSJSimulatedRirDataModule(
        rir_dataset=rir_dataset, dry_signal_target_len=49151, proportion_val_rir=0.0
    )
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()

    y, (x, h, rir_properties) = next(iter(loader))
    # h = h[0]

    if False:
        rt_60 = 0.2
        sigma = 0.1
        h = polack_analysis_synthesis.polack_synthesis(sigma, rt_60)

    h_hat_polack = polack_analysis_synthesis.training_step((y, (h, rir_properties)), batch_idx=0)["pred"]
    h_hat_bandwise = bandwise_polack_analysis_synthesis.training_step((y, (h, rir_properties)), batch_idx=0)["pred"]

    plt.close("all")
    fig_time, ax_ext = plt.subplots()
    ax_ext.plot(h_hat_polack[0].squeeze(), label="polack")
    ax_ext.plot(h_hat_bandwise[0].squeeze(), label="bandwise")
    ax_ext.plot(h[0].squeeze(), label="tgt")
    ax_ext.legend()
    fig_time.show()

    h_late = early_echoes_masking_module.rir_to_late(h.clone(), {}, include_direct_path=False)
    h_late_scaled = h_late / h_late.abs().square().sum(axis=-1)
    h_hat_polack_late = early_echoes_masking_module.rir_to_late(h_hat_polack.clone(), {}, include_direct_path=False)
    h_hat_polack_late_scaled = h_hat_polack_late / h_hat_polack_late.abs().square().sum(axis=-1)
    h_hat_bandwise_late = early_echoes_masking_module.rir_to_late(h_hat_bandwise.clone(), {}, include_direct_path=False)
    h_hat_bandwise_late_scaled = h_hat_bandwise_late / h_hat_bandwise_late.abs().square().sum(axis=-1)

    fig_tf, (ax_tf_oracle, ax_tf_polack, ax_tf_bandwise) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax_tf_oracle.set_title("ground-truth")
    spec_oracle = bandwise_polack_analysis_synthesis.stft_module(h).abs().square().log()
    spec_oracle_sacaled = spec_oracle / spec_oracle.max()
    ax_tf_oracle.imshow(spec_oracle_sacaled[0].squeeze(), origin="lower")
    ax_tf_polack.set_title("Polack")
    spec_polack = bandwise_polack_analysis_synthesis.stft_module(h_hat_polack).abs().square().log()
    spec_polack_scaled = spec_polack / spec_polack.max()
    ax_tf_polack.imshow(spec_polack_scaled[0].squeeze(), origin="lower")
    # spec, freqs, t, im = ax_tf_polack.specgram(h_hat_polack_late_scaled[0].squeeze(), NFFT=512, Fs=16000)
    ax_tf_bandwise.set_title("Bandwise-Polack")
    # spec, freqs, t, im = ax_tf_bandwise.specgram(h_hat_bandwise_late_scaled[0].squeeze(), NFFT=512, Fs=16000)
    spec_bandwise = bandwise_polack_analysis_synthesis.stft_module(h_hat_bandwise).abs().square().log()
    spec_bandwise_scaled = spec_bandwise / spec_bandwise.max()
    ax_tf_bandwise.imshow(spec_bandwise_scaled[0].squeeze(), origin="lower")
    fig_tf.show()
    # breakpoint()



def measure_mean_and_std_over_dataset(
    rir_root: str = "./data/rirs_v2",
    intersect_zero: bool = False,
    normalize_energy: bool = False,
):

    import numpy as np
    from datasets import SynthethicRirDataset, WSJ1SimulatedRirDataModule, EarsOnlyRIRsDataset
    import tqdm.auto as tqdm
    import matplotlib.pyplot as plt
    from model.utils.tensor_ops import crop_or_zero_pad_to_target_len
    import datetime
    import os

    device = torch.device("cuda")
    res_all_parameters = []
    polack_analysis_kwargs = dict(
        regress=True,
        intersect_zero=intersect_zero,
        regression_max_energy=-5,
        regression_min_energy=-25,
        rt_60_depends_from_slope_only=True,
        direct_path_duration_ms=2.5,
    )
    if "rirs_v2" in rir_root:
        rir_dataset = SynthethicRirDataset(rir_root)
        dataset_name = "simulated"
        polack_analysis_kwargs["rir_length"] = 32000
        polack_analysis_kwargs["fs"] = 16000

    elif "ears" in rir_root.lower():
        dataset_name = "EARS"
        rir_dataset = EarsOnlyRIRsDataset(rir_root, fs=48000)
        polack_analysis_kwargs["rir_length"] = 192000
        polack_analysis_kwargs["fs"] = 48000
    else:
        raise NotImplementedError()
    model = PolackAnalysis(**polack_analysis_kwargs).to(device=device)
    parameters_str = f"(dataset={dataset_name})"
    if intersect_zero or normalize_energy:
        parameters_str = (
            parameters_str[:-1] + f", intersect_zero={intersect_zero}, normalize_energy={normalize_energy})"
        )
    res = {"sigma": np.zeros(len(rir_dataset)), "rt_60": np.zeros(len(rir_dataset))}
    for i, (rir, d) in enumerate(tqdm.tqdm(rir_dataset)):
        # model.plot = i == 0
        peak_index = torch.argmax(torch.abs(rir))
        rir_peak = rir[..., peak_index]
        rir_aligned = rir[..., peak_index:]
        rir_aligned_scaled = rir_aligned / rir.abs()[..., peak_index, None]
        if normalize_energy:
            rir_aligned_scaled /= rir_aligned_scaled.abs().square().sum(-1, keepdim=True).sqrt()
        rir_aligned_scaled_cropped = crop_or_zero_pad_to_target_len(
            rir_aligned_scaled, target_len=polack_analysis_kwargs["rir_length"]
        )

        sigma, rt_60 = model(rir_aligned_scaled_cropped.to(device=device))
        if not normalize_energy and sigma > 1.0:
            print("Inconsistent sigma found at rir \n" + f"{rir_dataset.rir_list[i]}" + "\n RIR might be clipping")
            plt.figure()
            plt.plot(rir_aligned_scaled.squeeze())
            plt.show()
            res["sigma"][i] = np.nan
            res["rt_60"][i] = np.nan
        else:
            res["sigma"][i] = sigma.cpu()
            res["rt_60"][i] = rt_60.cpu()

    # print("mean", res.mean())
    # print("median", np.median(res))
    # print("std", np.std(res))
    os.makedirs("res/" + str(datetime.date.today()) + "/distributions_of_polack_corrected", exist_ok=True)
    plt.figure()
    plt.suptitle(r"Distribution of $\sigma$" + "\n" + parameters_str)
    plt.hist(res["sigma"], 100)
    plt.savefig(
        "res/" + str(datetime.date.today()) + "/distributions_of_polack_corrected/sigma" + parameters_str + ".pdf"
    )

    plt.figure()
    plt.suptitle(r"Distribution of $RT_{60}$" + "\n" + parameters_str)
    plt.hist(res["rt_60"], 100)
    plt.savefig(
        "res/" + str(datetime.date.today()) + "/distributions_of_polack_corrected/rt_60" + parameters_str + ".pdf"
    )
    return res


def plot_hist_of_parameters(also_plot_influence_of_normalization_and_intersect_zero=False):
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime

    plt.close("all")

    res_grouped_by_dataset = dict()
    for dataset_path, intersect_zero, normalize_energy in itertools.product(
        (
            "./data/speech/EARS-Reverb/train_unique_rirs/",
            "./data/rirs_v2",
        ),
        (True, False) if also_plot_influence_of_normalization_and_intersect_zero else (False,),
        (True, False) if also_plot_influence_of_normalization_and_intersect_zero else (False,),
    ):

        res_current_config = measure_mean_and_std_over_dataset(
            rir_root=dataset_path,
            intersect_zero=intersect_zero,
            normalize_energy=normalize_energy,
        )
        if dataset_path not in res_grouped_by_dataset.keys():
            res_grouped_by_dataset[dataset_path] = dict(sigma=dict(), rt_60=dict())

        res_sigma, res_rt_60 = res_current_config["sigma"], res_current_config["rt_60"]
        res_grouped_by_dataset[dataset_path]["sigma"][(intersect_zero, normalize_energy)] = res_sigma
        res_grouped_by_dataset[dataset_path]["rt_60"][(intersect_zero, normalize_energy)] = res_rt_60
    for dataset_path in res_grouped_by_dataset:
        if "rirs_v2" in dataset_path:
            dataset_name = "Simulated RIRs"
        elif "ears" in dataset_path.lower():
            dataset_name = "RIRs from EARS-Reverb"
        else:
            dataset_name = dataset_path
        fig, (ax_rt_60, ax_sigma, ax_scatter) = plt.subplots(1, 3, figsize=(30, 10))
        fig.suptitle(r"Distributions of Polack's $\sigma$ and $RT_{60}$ " + f"for dataset of {dataset_name}.pdf")
        res_sigma_list = []
        res_rt_60_list = []
        labels = []
        for intersect_zero in (True, False):
            for normalize_energy in (True, False):
                labels.append(f"intersect_zero={intersect_zero}, normalize_energy={normalize_energy}")
                try:
                    res_sigma_list.append(
                        res_grouped_by_dataset[dataset_path]["sigma"][(intersect_zero, normalize_energy)]
                    )
                    res_rt_60_list.append(
                        res_grouped_by_dataset[dataset_path]["rt_60"][(intersect_zero, normalize_energy)]
                    )
                except KeyError:
                    # the combination of intersect_zero and normalize_energy is not found
                    pass
        res_rt_60_array = np.stack(res_rt_60_list, -1)
        res_sigma_array = np.stack(res_sigma_list, -1)
        bins = 20
        ax_rt_60.hist(res_rt_60_array, bins=bins, histtype="bar", label=labels)
        ax_rt_60.set_title(r"$RT_{60}$")
        ax_sigma.set_title(r"$\sigma$")
        ax_sigma.hist(
            res_sigma_array,
            bins=bins,
            histtype="bar",
            label=labels,
        )
        ax_scatter.plot(res_sigma_array, res_rt_60_array, ".", label=labels)
        ax_scatter.set_xlabel(r"$\sigma$")
        ax_scatter.set_ylabel(r"$RT_{60}$")
        if also_plot_influence_of_normalization_and_intersect_zero:
            ax_sigma.legend()
            ax_scatter.legend()
            ax_rt_60.legend()
        fig.savefig(
            f"res/{str(datetime.date.today())}/distributions_of_polack_corrected/distributions_sigma_rt_60_{dataset_name}.pdf"
        )
    return res_grouped_by_dataset


# %% Main

if __name__ == "__main__":
    # res_grouped_by_dataset = plot_hist_of_parameters()

    # test_polack_prego()
    res = plot_hist_of_parameters()
    # test_polack()
    # test_stft_polack(use_GL=True)
    # test_stft_polack(use_GL=False)
    # test_stft_polack(use_GL=False, resample_rir=False)
    # test_stft_polack(use_GL=True, resample_rir=False)
    # test_rt_60_divergence_with_oracle_properties(normalization_method="peak")
    # res = test_polack_analysis_synthesis(num_batches=1, plot_examples=False, device="cuda", num_optim_steps=1)
    # # test_bandwise_polack()
    # measure_energy_deviation_after_noise_floor(measure_energy_divergence_below=-35)
    # measure_energy_deviation_after_noise_floor(measure_energy_divergence_below=-35, direct_path_duration_ms=2.5)
    # measure_energy_deviation_after_noise_floor(
    # measure_energy_divergence_below=-25
    # )
    # measure_energy_deviation_after_noise_floor(measure_energy_divergence_below=-25, direct_path_duration_ms=2.5)
