#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 23:49:09 2025

@author: louis
"""

# sshfs work: mnt/

import os
from tqdm.auto import tqdm
from datasets import WSJ1Dataset
import speechbrain as sb
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
import pandas as pd
import torch
import numpy as np
import torchaudio
import argparse

SAMPLERATE = 16000


def get_transcript(wv1_path):
    transcript_id = os.path.basename(wv1_path).split(".")[0]
    transcript_file = os.path.join(
        os.path.dirname(wv1_path), f"{transcript_id[:-2]}00.lsn"
    )
    with open(transcript_file, "r") as f:
        for line in f:
            if transcript_id.lower() in line.lower():
                transcript = line.split("(")[
                    0
                ].strip()  # Extract the transcription part before any metadata
    return transcript


def invert_dict(d):
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    return inv_map


def find_tensor_pairs(dict1, dict2):
    key_pairs = []

    for key1, tensor1 in dict1.items():
        for key2, tensor2 in dict2.items():
            if torch.allclose(tensor1, tensor2):
                key_pairs.append((key1, key2))
    return key_pairs


def associate_wav_to_wv1(
    audio_root="data/speech",
    subset="test",
):
    d_wv1 = WSJ1Dataset(
        audio_root,
        subset=subset,
        wav=False,
    )
    d_wav = WSJ1Dataset(
        audio_root,
        subset=subset,
        wav=True,
    )

    # first association by length
    file_to_length_wav = {
        f: sb.dataio.dataio.read_audio_info(f).num_frames
        for f in tqdm(d_wav.paths_list)
    }
    file_to_length_wv1 = {
        f: sb.dataio.dataio.read_audio_info(f).num_frames
        for f in tqdm(d_wv1.paths_list)
    }

    length_to_wv1 = invert_dict(file_to_length_wv1)
    length_to_wav = invert_dict(file_to_length_wav)

    # smallest_of_two_sets=file_to_length_wav if len(file_to_length_wav) < len(file_to_length_wv1) else file_to_length_wv1
    pairs = []
    for length, wv1_files in length_to_wv1.items():
        wav_files = length_to_wav[length]
        # Easy case, we have a single file for the length
        if len(wv1_files) == 1:
            wv1_file = wv1_files[0]
            if len(wav_files) == 1:
                wav_file = wav_files[0]
                pairs.append((wav_file, wv1_file))
            else:
                raise RuntimeError()
        else:  # if several files are of the same length
            wv1_tensors = {
                f: sb.dataio.dataio.read_audio(f) for f in wv1_files
            }
            wav_tensors = {
                f: sb.dataio.dataio.read_audio(f) for f in wav_files
            }
            if len(wv1_tensors) != len(wav_tensors):
                raise RuntimeError()
            pairs.extend(find_tensor_pairs(wav_tensors, wv1_tensors))
    return pairs


def export_transcript_csv(
    paired_filenames,
    save_path="data/speech/WSJ/WSJ1_wav_mic1/test_transcripts.csv",
):
    wav_paths, wv1_paths = zip(*paired_filenames)
    transcripts = [get_transcript(f) for f in tqdm(wv1_paths)]
    df = pd.DataFrame(
        {"wv1": wv1_paths, "wav": wav_paths, "transcript": transcripts}
    )
    df.set_index("wav", inplace=True)
    df.to_csv(save_path)
    return df


def eval_dataset(
    model,
    logs_path="lightning_logs",
    device="cuda",
    dataset_name="test_wsj_reverb",
    taps=10,
    delay=3,
    iterations=3,
):
    from model.utils.run_management import instantiate_model_only
    from model.utils.metrics import all_speech_metrics
    from model.speech_models.pytorch_wpe import wpe_one_iteration
    from torch_complex import ComplexTensor

    device = torch.device(device)
    metrics_folder = os.path.join(
        logs_path, "test", "asr", dataset_name, model
    )
    os.makedirs(metrics_folder, exist_ok=True)

    dataset_path = os.path.join("data", dataset_name)
    df = pd.read_csv(os.path.join(dataset_path, "properties.csv"))

    rel_lengths = torch.tensor([1.0], device=device)

    asr_models_and_wers = {
        k: (
            EncoderDecoderASR.from_hparams(
                source="speechbrain/" + k,
                savedir="pretrained_models/" + k,
                run_opts={"device": device},
            ),
            ErrorRateStats(),
        )
        for k in ("asr-crdnn-rnnlm-librispeech", "asr-wav2vec2-commonvoice-en")
    }

    from test_wpe import process_with_nara_wpe

    # first model
    if "bilstm_strong_100" in model.lower():
        full_model = (
            instantiate_model_only(
                os.path.join(
                    logs_path,
                    "taslp",
                    "further_bilstm_null_100",
                    "version_0",
                    "config.yaml",
                )
            )
            .to(device=device)
            .eval()
        )
    elif "bilstm_strong" in model.lower():
        full_model = instantiate_model_only(
            os.path.join(
                logs_path,
                "taslp",
                "supervised",
                "ears_bilstm_null",
                "version_0",
                "config.yaml",
            )
        ).to(device=device).eval()

    elif "bilstm_unsupervised" in model.lower():
        full_model = (
            instantiate_model_only(
                os.path.join(
                    logs_path,
                    "taslp",
                    "unsupervised",
                    "ears_bilstm_null_100",
                    "version_1",
                    "config.yaml",
                )
            )
            .to(device=device)
            .eval()
        )

    all_speech_metrics = {
        str(m): m.to(device=device) for m in all_speech_metrics
    }
    res_dict = {
        k: torch.full((len(df),), torch.nan, device=device)
        for k in all_speech_metrics.keys()
    }

    for audio_idx, (transcript, rt_60) in tqdm(
        df[["transcript", "rt_60"]].iterrows(), total=len(df)
    ):
        if audio_idx > float("inf"):
            break
        y, fs = torchaudio.load(
            os.path.join(dataset_path, "wet", f"{audio_idx}.wav")
        )
        s, fs = torchaudio.load(
            os.path.join(dataset_path, "dry", f"{audio_idx}.wav")
        )

        if "reverberant" in model.lower() or "wet" in model.lower():
            s_hat = y
        elif "dry" in model.lower():
            s_hat = s
        elif "nara_wpe" in model.lower():
            s_hat = process_with_nara_wpe(
                y, taps=taps, delay=delay, iterations=iterations
            )
        elif "dnn_wpe" in model.lower() and "bilstm" in model.lower():
            noisy_stft = full_model.speech_model.stft_module(
                y.unsqueeze(0).to(device=device)
            )  # B, C, F, T
            noisy_spec = noisy_stft.abs().squeeze(-3).transpose(-1, -2)
            with torch.no_grad():
                mask = full_model.speech_model.model(
                    noisy_spec,
                    lengths=torch.ones(
                        noisy_stft.size(0), device=noisy_spec.device
                    ),
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
            try:
                predict_stft_after_wpe = wpe_one_iteration(
                    predict_stft_wpe_format,
                    power_spec_wpe_format,
                    taps=taps,
                    delay=delay,
                )
            except torch._C._LinAlgError:
                predict_stft_after_wpe=predict_stft_wpe_format
            # go back to time domain
            predict_stft = (
                predict_stft_after_wpe.real + 1j * predict_stft_after_wpe.imag
            ).transpose(-2, -3)
            s_hat = full_model.speech_model.istft_module(predict_stft)[0]
        else:
            with torch.no_grad():
                s_hat = full_model.predict_dry_speech(
                    y.to(device=device).unsqueeze(0)
                ).squeeze(0)
        # eval WER
        for asr_model, wer_stats in asr_models_and_wers.values():
            estimated_transcript = asr_model.transcribe_batch(
                s_hat.to(device=device), rel_lengths
            )[0][0]
            wer_stats.append([audio_idx], [estimated_transcript], [transcript])
        # eval metrics
        for metric_name, metric_module in all_speech_metrics.items():
            res_dict[metric_name][audio_idx] = metric_module(
                s_hat[..., : s.shape[-1]].to(device=device),
                s.to(device=device),
            )

    for metric_name, metrics_tensor in res_dict.items():
        np.save(
            os.path.join(
                metrics_folder,
                "".join(
                    ki for ki in metric_name if (ki.isalnum() or ki == "_")
                )
                + ".npy",
            ),
            metrics_tensor.detach().cpu().numpy(),
        )
    with open(os.path.join(metrics_folder, "wer.txt"), "w") as f:
        for asr_model_name, (_, wer_stats) in asr_models_and_wers.items():
            print(f"{asr_model_name}: {wer_stats.summarize()['WER']}", file=f)


def parse_args():
    parser = argparse.ArgumentParser(description="Test WPE")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset_name", type=str, default="test_wsj_reverb")
    parser.add_argument("--logs_path", type=str, default="lightning_logs")
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--taps", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    eval_dataset(**vars(args))
#     paired_filenames = associate_wav_to_wv1()
#     export_transcript_csv(paired_filenames)

# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
