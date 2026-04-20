#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dereverberation script to process a file"""

import os
import torch
import torchaudio
import argparse
import contextlib
import glob
import warnings

with contextlib.redirect_stdout(None):
    from model.utils.run_management import instantiate_model_only

LOG_ROOT = "UDREAM_checkpoints"

# %%


def test_load_model():
    model = instantiate_model_only(
        config_path=f"./{LOG_ROOT}/supervised/ears_tflocoformer_null/version_0/config.yaml",
    )


def test_load_all_models():
    import glob

    config_paths = (
        glob.glob(
            os.path.join(f"./{LOG_ROOT}", "supervised", "**", "config.yaml"),
            recursive=True,
        )
        + glob.glob(
            os.path.join(f"./{LOG_ROOT}", "weak", "**", "config.yaml"),
            recursive=True,
        )
        + glob.glob(
            os.path.join(f"./{LOG_ROOT}", "unsupervised", "**", "config.yaml"),
            recursive=True,
        )
    )
    for config_path in config_paths:
        model = instantiate_model_only(
            # config_path="remote_logs/old_logs/FullSubNet_dry_wsj1/version_4_test_wsj1_nonoise/version_0/config.yaml",
            config_path=config_path,
        )


# %%


def model_predict_wav(
    model,
    input_audio="./examples/wet.wav",
    output_audio=None,
    use_cuda_if_available: bool = True,
):
    if output_audio is None:
        output_audio = os.path.join(
            os.path.dirname(input_audio),
            "".join(os.path.basename(input_audio).split(".")[:-1]) + "_predicted_dry.wav",
        )
    if use_cuda_if_available and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    y, fs = torchaudio.load(input_audio)
    if model.speech_model.fs != fs:
        raise ValueError(f"Expected audio samplerate to be {model.speech_model.fs} Hz, input audio is at {fs} Hz")
    y = y.to(device=device)
    model.freeze()
    model.eval()
    model = model.to(device=device)
    s = model.predict_dry_speech(y[None, ...])
    s /= s.abs().max()
    print(f"saving audio to {output_audio}")
    torchaudio.save(output_audio, s[0, ...].cpu(), sample_rate=fs)
    return s


def get_log_path(model_variant, supervision_scenario, supervision_variant=None, dataset="ears"):
    # model variant
    if any(mv in model_variant.lower() for mv in ("fsn", "fullsubnet")):
        model_title = "FullSubNet (FSN)"
        model_path = "FSN"
    elif any(mv in model_variant.lower() for mv in ("tfl", "locoformer")):
        model_title = "TF-Locoformer (TFL)"
        model_path = "tflocoformer"
    elif any(mv in model_variant.lower() for mv in ("bi_lstm", "bilstm", "bi-lstm")):
        model_title = "BiLSTM"
        model_path = "bilstm"
    else:
        raise ValueError("Model variant not supported")

    # supervision_scenario
    if "strong" in supervision_scenario.lower():
        supervision_title = "with strong supervision"
        supervision_path = "supervised"
        default_supervision_variant = "null"
    elif "weak" in supervision_scenario.lower():
        supervision_title = "with weak supervision of only RT60 and DRR"
        supervision_path = "weak"
        default_supervision_variant = "1draw"
    elif any(sv in supervision_scenario.lower() for sv in ("none", "auto", "unsupervised")):
        supervision_title = "in an unsupervised setting using a pretrained reverberation model"
        supervision_path = "unsupervised"
        default_supervision_variant = "null_100"
    else:
        raise ValueError("Supervision scenario not supported")
    if supervision_variant is not None:
        raise NotImplementedError("Supervision_variant not yet supported")

    # dataset
    if "ears" in dataset.lower():
        dataset_path = "ears"
    elif "synth" in dataset.lower():
        dataset_path = "synthethic"
    else:
        raise ValueError(f"Dataset {dataset} is not supported")

    print(f"Using {model_title} trained on {dataset} {supervision_title}")
    return os.path.join(LOG_ROOT, supervision_path, dataset_path + "_" + model_path + "_" + default_supervision_variant)


def get_config_file(log_path):
    config_files = glob.glob(os.path.join(log_path, "**", "config.yaml"), recursive=True)
    if len(config_files) != 1:
        raise RuntimeError("Expected one and only one config file")
    return config_files[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Dereverberate an audio file")
    parser.add_argument("input_audio", help="input (reverberant) audio path", type=str)
    supervision_scenario_help = (
        "Supervision scenario, "
        + "must be either 'strong' (Wet/Dry pairs), "
        + "'weak' (reverberation parameters)"
        + "or 'unsupervised' (guided by a reverberation modeltrained usin 100 audio examples"
    )
    parser.add_argument(
        "--supervision_scenario",
        "-s",
        help=f"{supervision_scenario_help}. Default: unsupervised",
        type=str,
        default="unsupervised",
    )
    parser.add_argument(
        "--model_variant",
        "-m",
        help="Model variant: TF-Locoformer, FullSubNet or BiLSTM. Default: BiLSTM",
        type=str,
        default="BiLSTM",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Dataset the model has been trained on: EARS or Synthetic. Default: EARS",
        type=str,
        default="EARS",
    )
    parser.add_argument(
        "--use_cuda_if_available",
        "--cuda",
        help="Use CUDA if CUDA accelerator is available.",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--output_audio",
        "-o",
        help="output (predicted dry) audio path. Default: '[INPUT_AUDIO]_predicted_dry.wav'",
        default=None,
    )
    args = parser.parse_args()
    return args


def dereverberate_audio(
    input_audio,
    model_variant="bilstm",
    supervision_scenario="unsupervised",
    dataset="EARS",
    output_audio=None,
    use_cuda_if_available: bool = True,
):
    if not os.path.exists(input_audio):
        raise ValueError(f"input audio not found {input_audio}")
    log_path = get_log_path(model_variant=model_variant, supervision_scenario=supervision_scenario, dataset=dataset)
    if not os.path.isdir(log_path):
        raise NotImplementedError(f"Checkpoint not found in {log_path}")
    config_file = get_config_file(log_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = instantiate_model_only(config_file)
        model_predict_wav(
            model,
            input_audio=input_audio,
            output_audio=output_audio,
            use_cuda_if_available=use_cuda_if_available,
        )


if __name__ == "__main__":
    args = parse_args()
    dereverberate_audio(**vars(args))


# tast_load_all_model()
