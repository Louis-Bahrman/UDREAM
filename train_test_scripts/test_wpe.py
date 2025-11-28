#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import os
import tqdm.auto as tqdm

from nara_wpe.wpe import wpe as nara_wpe
from nara_wpe.utils import stft, istft
from espnet2.enh.layers.wpe import wpe as espnet_wpe

from model.speech_models.pytorch_wpe import wpe_one_iteration
from torch_complex import ComplexTensor
import argparse

stft_options = dict(size=512, shift=128)


from datasets import (
    EARSReverbDataModule,
    PairedDataModule,
)
from model.utils.metrics import all_speech_metrics

data_modules = {
    "ears": EARSReverbDataModule(
        enable_caching_train=False, enable_caching_val=False, batch_size=1
    ),
    "synth": PairedDataModule(path="./data/test_ears_same"),
}

wpe_implementations = {"espnet": espnet_wpe, "nara": nara_wpe}
channels = 8
sampling_rate = 16000
delay = 3
iterations = 5
taps = 10
alpha = 0.9999


speech_metrics = dict(
    zip(["SISDR", "ESTOI", "WB-PESQ", "SRMR"], all_speech_metrics)
)


def test_wpe(
    dataset_name,
    wpe_implementation_name,
    delay=delay,
    iterations=iterations,
    taps=taps,
    device="cuda",  # only for DNN
):
    datamodule = data_modules[dataset_name]
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.test_dataloader()

    res_dict = {
        k: np.full((len(datamodule.dataset_test)), np.nan)
        for k in speech_metrics
    }

    # first model
    if "dnn_wpe" in wpe_implementation_name.lower():
        from model.utils.run_management import instantiate_model_only

        if "bilstm_strong_100" in wpe_implementation_name.lower():
            full_model = instantiate_model_only(
                os.path.join(
                    "lightning_logs",
                    "taslp",
                    "further_bilstm_null_100",
                    "version_0",
                    "config.yaml",
                )
            )
        elif "bilstm_unsupervised" in wpe_implementation_name.lower():
            full_model = instantiate_model_only(
                os.path.join(
                    "lightning_logs",
                    "taslp",
                    "unsupervised",
                    "ears_bilstm_null_100",
                    "version_1",
                    "config.yaml",
                )
            )
        elif "bilstm_strong_5percent" in wpe_implementation_name.lower():
            full_model = instantiate_model_only(
                os.path.join(
                    "lightning_logs",
                    "taslp",
                    "taslp_marius_ablations_phase_and_strong",
                    "ears16_reduced-5%_BiLSTM_strong",
                    "version_1",
                    "config.yaml",
                )
            )
        elif "bilstm_strong" in wpe_implementation_name.lower():
            full_model = instantiate_model_only(
                os.path.join(
                    "lightning_logs",
                    "taslp",
                    "supervised",
                    "ears_bilstm_null",
                    "version_0",
                    "config.yaml",
                )
            )
        else:
            raise NotImplementedError()
        full_model = full_model.to(device=device).eval()

    for batch_idx, (y, (s, _, _)) in enumerate(tqdm.tqdm(dataloader)):
        if "nara" in wpe_implementation_name.lower():
            y = y.squeeze().unsqueeze(0).numpy()

            Y = stft(y, **stft_options).transpose(2, 0, 1)
            Z = nara_wpe(
                Y,
                taps=taps,
                delay=delay,
                iterations=iterations,
                statistics_mode="full",
            ).transpose(1, 2, 0)
            z = istft(
                Z, size=stft_options["size"], shift=stft_options["shift"]
            )
            s_hat = torch.as_tensor(z)
        elif "espnet" in wpe_implementation_name.lower():
            Y = torch.stft(
                y[0],
                n_fft=stft_options["size"],
                hop_length=stft_options["shift"],
                window=torch.hann_window(stft_options["size"]),
                return_complex=True,
            ).permute(2, 0, 1)
            S_hat = espnet_wpe(
                Y, taps=taps, delay=delay, iterations=iterations
            )
            s_hat = torch.istft(
                S_hat.permute(1, 2, 0),
                n_fft=stft_options["size"],
                hop_length=stft_options["shift"],
                window=torch.hann_window(stft_options["size"]),
            )
        elif "dnn_wpe" in wpe_implementation_name.lower():
            s_hat = process_with_dnn_wpe(
                y.squeeze(0).to(device=device), taps, delay, full_model
            ).cpu()
        else:
            raise NotImplementedError()
        s_hat = s_hat[None, ..., : s.size(-1)]
        s = s[..., : s_hat.size(-1)]
        for metric_name, metric_module in speech_metrics.items():
            res_dict[metric_name][batch_idx] = metric_module(s_hat, s)
    save_path = os.path.join(
        "lightning_logs",
        "test",
        "wpe",
        f"{dataset_name}_{wpe_implementation_name}_{delay}_{iterations}_{taps}",
    )
    os.makedirs(save_path, exist_ok=True)
    for k, v in res_dict.items():
        np.save(os.path.join(save_path, k), v)
    print({k: (v.mean(), v.std()) for k, v in res_dict.items()})


def process_with_nara_wpe(y, taps, delay, iterations):
    y = y.squeeze().cpu().unsqueeze(0).numpy()

    Y = stft(y, **stft_options).transpose(2, 0, 1)
    Z = nara_wpe(
        Y,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode="full",
    ).transpose(1, 2, 0)
    z = istft(Z, size=stft_options["size"], shift=stft_options["shift"])
    s_hat = torch.as_tensor(z)
    return s_hat


def process_with_dnn_wpe(y, taps, delay, full_model):
    noisy_stft = full_model.speech_model.stft_module(
        y.unsqueeze(0).to(device=full_model.device)
    )  # B, C, F, T
    noisy_spec = noisy_stft.abs().squeeze(-3).transpose(-1, -2)
    with torch.no_grad():
        mask = full_model.speech_model.model(
            noisy_spec,
            lengths=torch.ones(noisy_stft.size(0), device=noisy_spec.device),
        )
    mask = mask.clamp(min=full_model.speech_model.min_mask)
    mask_reshaped = mask.unsqueeze(-3).transpose(-1, -2)
    predict_stft = torch.mul(mask_reshaped, noisy_stft)
    # now take the absolute value
    predict_spec = torch.abs(predict_stft)
    # reformat to dnn_wpe:bcft -> fct
    power_spec_wpe_format = predict_spec[..., 0, :, :]
    predict_stft_wpe_format = ComplexTensor(
        predict_stft.real, predict_stft.imag
    ).transpose(-2, -3)
    # apply dnn_wpe
    predict_stft_after_wpe = wpe_one_iteration(
        predict_stft_wpe_format,
        power_spec_wpe_format,
        taps=taps,
        delay=delay,
    )
    # go back to time domain
    predict_stft = (
        predict_stft_after_wpe.real + 1j * predict_stft_after_wpe.imag
    ).transpose(-2, -3)
    return full_model.speech_model.istft_module(predict_stft)[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Test WPE")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--implementation", type=str)
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--taps", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test_wpe(
        dataset_name=args.dataset,
        wpe_implementation_name=args.implementation,
        delay=args.delay,
        iterations=args.iterations,
        taps=args.taps,
    )
