#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 00:39:42 2024

@author: louis
"""
import os
import scipy
import numpy as np
import itertools
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import shutil
import functools
import seaborn as sns

from matplotlib.colors import LogNorm

rng = np.random.default_rng()

# %% Variables

save_dir = "figures"

speech_models = {"BiLSTM": "bilstm", "FSN": "fsn", "TFL": "tflocoformer"}
strong_supervision_variants = {"dry": "null", "RIR": "oraclerir"}
weak_supervision_variants = {
    "Single": "1draw",
    "Average": "10draws",
    "Best": "10drawsbestchannel",
}

reverb_model_supervisions = {
    "PM": "null",
    "RM": "rereverb",
}
reverb_model_fractions = {
    "100 examples": "100",
    r"5 percent": "0.05",
    "full dataset": "full",
}

# Perfect choice of values since they are uniform on log-scale
TRAINING_DATASET_TOTAL_SIZE = 32485
num_samples = np.array(
    [
        100,
        5 * TRAINING_DATASET_TOTAL_SIZE // 100,
        TRAINING_DATASET_TOTAL_SIZE,
    ]
)
fractions_to_samples = dict(zip(reverb_model_fractions.keys(), num_samples))

reverb_model_variants = {
    f"{k1} & {k2}": f"{v1}_{v2}"
    for k1, v1 in reverb_model_supervisions.items()
    for k2, v2 in reverb_model_fractions.items()
}

unsupervised_variants = {
    f"Reverb{k1} & {k2}": f"{v1}_{v2}"
    for k1, v1 in reverb_model_supervisions.items()
    for k2, v2 in reverb_model_fractions.items()
}

speech_supervision_types = {
    "strong": ("supervised", strong_supervision_variants),
    "weak": ("weak", weak_supervision_variants),
    "unsupervised": ("unsupervised", unsupervised_variants),
}

# train_test_datasets = {
#     "Trained and tested on synthetic RIRs": (
#         "synthethic",
#         "synth",
#     ),
#     "Trained on synthetic RIRs, tested on real RIRs": ("synthethic", "ears"),
#     "Trained and tested on real RIRs": ("ears", "ears"),
# }

train_test_datasets = {
    "Synthetic RIRs": (
        "synthethic",
        "synth",
    ),
    "Out-of-domain RIRs": ("synthethic", "ears"),
    "Real RIRs": ("ears", "ears"),
}

speech_metrics = {
    "SISDR": "val_dry_speech_model_ScaleInvariantSignalDistortionRatio0256.npy",
    "ESTOI": "val_dry_speech_model_ShortTimeObjectiveIntelligibility0256.npy",
    "WB-PESQ": "val_dry_speech_model_PerceptualEvaluationSpeechQuality0256.npy",
    "SRMR": "val_dry_speech_model_SRMRWrapper0256.npy",
}

input_speech_metrics = {
    k: v[:-4] + "_input.npy" for k, v in speech_metrics.items()
}

reverb_model_metrics = {
    "DRR MSE": "test_step_reverb_model_loss.npy",
    "Reverberation matching loss": "test_step_rereverberation_loss_loss.npy",
}

reverb_model_metrics = {
    "PM": "test_step_reverb_model_loss.npy",
    "RM": "test_step_rereverberation_loss_loss.npy",
}

# %% Gather results


def get_latest_version_path_containing_results(path):
    # distinct from the one in model/utils/run_management as it takes only the path that contains latest_results and not checkpoints
    if not os.path.exists(path):
        return os.path.join(path, "version_0")
    versions_paths = [
        os.path.abspath(os.path.join(path, version_name))
        for version_name in os.listdir(path)
    ]
    valid_versions_paths = [
        p for p in versions_paths if "latest_results" in os.listdir(p)
    ]
    return max(valid_versions_paths, key=os.path.getmtime)


def gather_reverberant_results():
    res = dict()
    for dataset_name, (
        train_set_path,
        test_set_path,
    ) in train_test_datasets.items():
        res[dataset_name] = dict()
        for metric_label, metric_val in input_speech_metrics.items():
            possible_paths = list(
                itertools.chain.from_iterable(
                    [
                        glob.glob(
                            os.path.join(
                                "lightning_logs",
                                "test",
                                speech_supervision_type,
                                f"{train_set_path}_*",
                                test_set_path,
                                "version_0",
                                "latest_results",
                                metric_val,
                            )
                        )
                        for speech_supervision_type, _ in speech_supervision_types.values()
                    ]
                )
            )
            path_1, path_2 = rng.choice(possible_paths, size=2, replace=False)
            if os.path.exists(path_1):
                array = np.load(path_1)
                if not (array == np.load(path_2)).all():
                    raise ValueError(
                        "Reverberant metrics should be consistent across models and supervision types"
                    )
            else:
                raise FileNotFoundError()
            res[dataset_name][metric_label] = array
    return res


def gather_results_speech_model():
    res = dict()
    for supervision_type_label, (
        supervision_type_path,
        supervision_variants,
    ) in speech_supervision_types.items():
        res[supervision_type_label] = dict()
        for model_name, model_path in speech_models.items():
            res[supervision_type_label][model_name] = dict()
            for (
                supervision_variant_label,
                supervision_variant_path,
            ) in supervision_variants.items():
                res[supervision_type_label][model_name][
                    supervision_variant_label
                ] = dict()
                for dataset_name, (
                    train_set_path,
                    test_set_path,
                ) in train_test_datasets.items():
                    res[supervision_type_label][model_name][
                        supervision_variant_label
                    ][dataset_name] = dict()
                    for metric_name, metric_val in speech_metrics.items():
                        if supervision_type_path == "supervised":
                            path = os.path.join(
                                "lightning_logs",
                                "test",
                                supervision_type_path,
                                f"{train_set_path}_{model_path}_{supervision_variant_path}",
                                test_set_path,
                                "version_0",
                                "latest_results",
                                metric_val,
                            )
                        else:
                            path = os.path.join(
                                get_latest_version_path_containing_results(
                                    os.path.join(
                                        "lightning_logs",
                                        "test",
                                        supervision_type_path,
                                        f"{train_set_path}_{model_path}_{supervision_variant_path}",
                                        test_set_path,
                                    )
                                ),
                                "latest_results",
                                metric_val,
                            )
                        if os.path.exists(path):
                            array = np.load(path)
                        else:
                            array = None
                        res[supervision_type_label][model_name][
                            supervision_variant_label
                        ][dataset_name][metric_name] = array
    return res


def gather_results_reverb_model():
    res = dict()
    for (
        reverb_model_supervision_label,
        reverb_model_supervision_path,
    ) in reverb_model_supervisions.items():
        res[reverb_model_supervision_label] = dict()
        for (
            reverb_model_fraction_label,
            reverb_model_fraction_path,
        ) in reverb_model_fractions.items():
            res[reverb_model_supervision_label][
                reverb_model_fraction_label
            ] = dict()
            for dataset_name, (
                train_set_path,
                test_set_path,
            ) in train_test_datasets.items():
                res[reverb_model_supervision_label][
                    reverb_model_fraction_label
                ][dataset_name] = dict()
                for metric_name, metric_val in reverb_model_metrics.items():
                    path = os.path.join(
                        get_latest_version_path_containing_results(
                            os.path.join(
                                "lightning_logs",
                                "test",
                                "reverb_model",
                                f"{train_set_path}_{reverb_model_supervision_path}_{reverb_model_fraction_path}",
                                test_set_path,
                            )
                        ),
                        "latest_results",
                        metric_val,
                    )
                    if os.path.exists(path):
                        array = np.load(path)
                    else:
                        array = None
                    res[reverb_model_supervision_label][
                        reverb_model_fraction_label
                    ][dataset_name][metric_name] = array
    return res


def gather_results_all_variants(model):
    if model not in ("wpe", "trainingless"):
        raise ValueError()
    import pandas as pd

    res = dict()
    model_dir = os.path.join("lightning_logs", "test", model)
    variant_dirs = sorted(os.listdir(model_dir))
    for variant_dir in variant_dirs:
        dataset = variant_dir.split("_")[0]
        if dataset not in res:
            res[dataset] = dict()
        if model == "wpe":
            start_variant_key = 1
        else:
            start_variant_key = 2
        variant_key = "_".join(variant_dir.split("_")[start_variant_key:])
        res[dataset][variant_key] = dict()
        for metric_label in speech_metrics:
            path = os.path.join(model_dir, variant_dir, metric_label + ".npy")
            res[dataset][variant_key][metric_label] = np.load(path)
    return res


gather_results_all_variants_wpe = functools.partial(
    gather_results_all_variants, model="wpe"
)
gather_results_all_variants_trainingless = functools.partial(
    gather_results_all_variants, model="trainingless"
)


def gather_results_best_variant(model, force_best_variant: str | None = None):
    res = gather_results_all_variants(model=model)
    res_synth = pd.DataFrame(res["synth"]).T
    res_ears = pd.DataFrame(res["ears"]).T
    joint_mean_res = res_synth.join(
        res_ears, lsuffix="_synth", rsuffix="_ears"
    ).map(np.mean)
    best_variant = joint_mean_res.idxmax().value_counts().idxmax()
    # Change label
    if force_best_variant:
        best_variant = force_best_variant
    print(best_variant)
    res_good_key = dict()
    for k, v in train_test_datasets.items():
        try:
            res_good_key[k] = res[v[1]][best_variant]
        except:
            res_good_key[k] = {m: np.array([]) for m in speech_metrics.keys()}
    return res_good_key


gather_results_best_variant_wpe = functools.partial(
    gather_results_best_variant, model="wpe"
)
gather_results_best_variant_trainingless = functools.partial(
    gather_results_best_variant, model="trainingless"
)


def gather_wers():
    res = {}
    wer_files = sorted(
        glob.glob(
            os.path.join(
                "lightning_logs",
                "test",
                "asr",
                "test_wsj_reverb",
                "**",
                "wer.txt",
            ),
            recursive=True,
        )
    )
    for file in wer_files:
        model = os.path.basename(os.path.dirname(file))
        res[model] = dict()
        with open(file) as f:
            for line in f.readlines():
                asr_model, asr_score = line.strip().split(": ")
                res[model][asr_model] = float(asr_score)
    prune_models_full_of_none(res)
    return pd.DataFrame(res).T


def gather_results_wsj():
    res = dict()
    path = os.path.join(
        "lightning_logs",
        "test",
        "asr",
        "test_wsj_reverb",
    )
    for model_name in os.listdir(path):
        res[model_name] = dict()
        for metric_key, metric_full_filename in speech_metrics.items():
            metric_filename = metric_full_filename[
                len("val_dry_speech_model_") :
            ]
            metric_path = os.path.join(path, model_name, metric_filename)
            try:
                array = np.load(metric_path)
            except:
                array = None
            res[model_name][metric_key] = array
    prune_models_full_of_none(res)
    return res


def gather_results_phase_invariant():
    res_dict = dict()
    dataset = list(train_test_datasets.keys())[-1]
    for model_key in list(speech_models.keys())[1:]:
        res_dict[model_key] = {"phaseinv": {dataset: dict()}}
        for metric_label, metric_filename in speech_metrics.items():
            path = os.path.join(
                get_latest_version_path_containing_results(
                    os.path.join(
                        "lightning_logs",
                        "test",
                        "taslp_marius_ablations_phase_and_strong",
                        f"ears16_PhaseInv{model_key}_weak",
                        "ears",
                    )
                ),
                "latest_results",
                metric_filename,
            )
            if os.path.exists(path):
                array = np.load(path)
            else:
                breakpoint()
                array = None
            res_dict[model_key]["phaseinv"][dataset][metric_label] = array
    return res_dict


def all_dict_is_none(d):
    if isinstance(d, dict):
        return all(all_dict_is_none(v) for v in d.values())
    return d is None


def prune_models_full_of_none(speech_results_dict):
    for supervision_type_label in speech_results_dict.keys():
        for model_label in list(
            speech_results_dict[supervision_type_label].keys()
        ):
            if all_dict_is_none(
                speech_results_dict[supervision_type_label][model_label]
            ):
                speech_results_dict[supervision_type_label].pop(model_label)


def all_keys(d):
    if isinstance(d, dict):
        return [list(d.keys())] + all_keys(next(iter(d.values())))
    return []


def flatten_dict(d, parent_key=()):
    """
    Recursively flattens a dictionary of arbitrary depth.

    Args:
    - d (dict): The dictionary to flatten.
    - parent_key (tuple): Keeps track of the parent keys (used for recursion).

    Returns:
    - dict: Flattened dictionary with tuple keys.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + (
            k,
        )  # Add the current key to the parent key tuple

        # if isinstance(v, dict) and isinstance(
        #     next(iter(v.values())), np.ndarray
        # ):
        #     items.extend((apply_to_layer(v, lambda a: dict(enumerate(a)), 1)))
        if isinstance(
            v, dict
        ):  # If the value is another dictionary, recurse into it
            items.extend(flatten_dict(v, new_key).items())
        # If the value is a numpy array, flatten it
        elif isinstance(v, np.ndarray):
            for i, val in enumerate(v):
                # Add index as part of the key
                items.append((new_key + (i,), val))
        else:
            items.append((new_key, v))  # For leaf nodes, just store the value
    return dict(items)


def dict_to_dataframe(d, column_names, wide_metrics=True):
    """
    Convert a flattened dictionary to a DataFrame with specified column names.

    Args:
    - flattened_dict (dict): Flattened dictionary with tuple keys.
    - column_names (list): List of column names to assign to the DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with the appropriate columns and values.
    """
    # Convert the flattened dictionary to a DataFrame
    flattened_dict = flatten_dict(d)
    df = pd.DataFrame.from_dict(
        flattened_dict, orient="index", columns=["value"]
    )

    # Convert tuple index into separate columns
    df.index = pd.MultiIndex.from_tuples(
        df.index, names=column_names + ["idx_in_dataset"]
    )
    df.reset_index(inplace=True)
    if "metric" in column_names and wide_metrics:
        new_index = [cn for cn in column_names if cn != "metric"] + [
            "idx_in_dataset"
        ]
        df = df.pivot(
            columns="metric",
            index=new_index,
            values="value",
        )
        df.reset_index(inplace=True)

    return df


def gather_results_strong_supervision_smaller_sizes():
    res_dict = dict()
    reverb_model_fractions_marius_notations = dict(
        zip(reverb_model_fractions.keys(), ["100ex", "5%", "full"])
    )
    dataset = list(train_test_datasets.keys())[-1]
    for model_key in list(speech_models.keys())[:2]:
        res_dict[model_key] = dict()
        for (
            dataset_size_label,
            dataset_name_in_path,
        ) in reverb_model_fractions_marius_notations.items():
            res_dict[model_key][dataset_size_label] = {dataset: dict()}
            for metric_label, metric_filename in speech_metrics.items():
                path = os.path.join(
                    get_latest_version_path_containing_results(
                        os.path.join(
                            "lightning_logs",
                            "test",
                            "taslp_marius_ablations_phase_and_strong",
                            f"ears16_reduced-{dataset_name_in_path}_{model_key}_strong",
                            "ears",
                        )
                    ),
                    "latest_results",
                    metric_filename,
                )
                if os.path.exists(path):
                    array = np.load(path)
                else:
                    array = None
                res_dict[model_key][dataset_size_label][dataset][
                    metric_label
                ] = array
    return res_dict


def create_weak_dataframe(speech_results_dict):
    # move metrics column to last position
    d = move_last_level_to_position(speech_results_dict["weak"], 0)
    # apply unfolding of np array
    d = apply_to_layer(d, lambda a: dict(enumerate(a)), depth(d))
    df2 = pd.json_normalize([{"metric": k, **v} for k, v in d.items()]).T


def nested_diff(d1, d2):
    if depth(d1) < depth(d2):
        raise NotImplementedError()
    # d recurse
    if isinstance(d1, dict) and isinstance(d2, dict):
        # if common
        if d1.keys() == d2.keys():
            return {
                k1: nested_diff(v1, v2)
                for (k1, v1), (k2, v2) in zip(d1.items(), d2.items())
            }
        else:
            # only recurse through d1
            return {k1: nested_diff(v1, d2) for k1, v1 in d1.items()}
    return d1 - d2


def is_well_indexed(df, unique_column_name="value"):
    first_cols = [
        c for c in df.columns if c != unique_column_name
    ]  # all columns except "value"

    # Count how many distinct values exist per group
    violations = (
        df.groupby(list(first_cols))["value"]
        .nunique()
        .reset_index(name="n_unique_values")
        .query("n_unique_values > 1")
    )
    return violations.empty


# def nested_diff(d1, d2):
#     # If both are dicts, recurse into keys
#     if isinstance(d1, dict) and isinstance(d2, dict):
#         result = {}
#         all_keys = set(d1.keys()) | set(d2.keys())
#         for k in all_keys:
#             result[k] = nested_diff(d1.get(k), d2.get(k))
#         return result

#     # If one is dict but the other is value -> descend into the dict
#     if isinstance(d1, dict) and not isinstance(d2, dict):
#         return {k: nested_diff(v, d2) for k, v in d1.items()}

#     if isinstance(d2, dict) and not isinstance(d1, dict):
#         return {k: nested_diff(d1, v) for k, v in d2.items()}

#     # Both are values (leaf nodes) → subtract
#     if d1 is None:
#         return -d2
#     if d2 is None:
#         return d1
#     return d1 - d2

# %% Make table and plots


def delta_label_for_plot(metric_key):
    delta_label = r"$\Delta$ " + metric_key
    if "SISDR" in metric_key:
        return delta_label + " (dB)"
    return delta_label


def format_array(results_array, bold=False, color=""):
    color = color or "black"
    return (
        (r"\bfseries " if bold else "")
        + (r"\textcolor{" + color + "}{")
        + r"\tablenum{"
        + f"{results_array.mean():.3f}"
        + r"}"
        + r"} & "
        + (r"\bfseries " if bold else "")
        + (r"\textcolor{" + color + "}{")
        + r"\tablenum{"
        + f"{results_array.std():.3f}"
        + r"}"
        + "}"
    )


# def format_array(results_array, bold=False, color=""):
#     color = color or "black"
#     return (
#         (r"\bfseries " if bold else "")
#         + (r"{")
#         + r"\tablenum{"
#         + f"{results_array.mean():.3f}"
#         + r"}"
#         + r"} & "
#         + (r"\bfseries " if bold else "")
#         + (r"{")
#         + r"\tablenum{"
#         + f"{results_array.std():.3f}"
#         + r"}"
#         + "}"
#     )


def line_of_results(results_dict, line_name, supervision_type_label):
    res_line = (
        r"\multicolumn{"
        # + str(3 if supervision_type_label == "unsupervised" else 2)
        + str(2)
        + r"}{r|}{"
        + line_name
        + "}"
    )
    for (
        dataset_label,
        dataset_results,
    ) in results_dict.items():
        for metric_label, metric_result in dataset_results.items():
            res_line += " & "
            res_line += format_array(metric_result)
    res_line += r" \\"
    return res_line


def make_table(
    speech_results_dict,
    wpe_results_dict,
    reverberant_results_dict,
    trainingless_results_dict=None,
    supervision_type_label="strong",
    merge_datasets: bool = True,
    bold_mask=None,
):
    if wpe_results_dict:
        comparison_with_baseline = compare_models_with_baseline(
            speech_results_dict, wpe_results_dict, alpha=0.001
        )
    supervision_type_caption = supervision_type_label
    if not merge_datasets:
        raise NotImplementedError()
        # or else we need to invert some loops and it's more difficult
    first_columns_declaration = r"l | c |"

    # Make header
    before_begin_tabular = (
        r"""\begin{table*}[t]
    \centering
    \caption{"""
        + supervision_type_caption
        + r""" \\Dereverberation scores $\pm$ standard deviation\\
            (for each metric, the higher the better.)
  }
  \sisetup{
detect-weight, %
mode=text, %
tight-spacing=true,
round-mode=places,
round-precision=2,
table-format=1.2,
}
\resizebox{\linewidth}{!}{
\setlength{\tabcolsep}{3pt}
"""
    )
    begin_tabular = (
        r"""\begin{NiceTabular}{"""
        + first_columns_declaration
        + r"*{"
        + str(len(train_test_datasets))
        + r"""}{S[round-precision=1,table-format=-2.1]@{\,\( \pm \)\,}S[round-precision=1,table-format=1.1]S@{\,\( \pm \)\,}SS@{\,\( \pm \)\,}S"""
    )
    if "SRMR" in speech_metrics:
        begin_tabular += r"""S[round-precision=1,table-format=1.1]@{\,\( \pm \)\,}S[round-precision=1,table-format=1.1]"""
    begin_tabular += r"}}"
    table_header_datasets = (
        " & ".join(
            ["", ""]
            + [
                r"\multicolumn{"
                + str(2 * len(speech_metrics))
                + r"}{c}{"
                + datasets_label
                + "}"
                for datasets_label in train_test_datasets.keys()
            ]
        )
        + r"\\"
    )
    table_header_metrics = (
        " & ".join(
            (
                ["target", "fraction"]
                if supervision_type_label == "unsupervised"
                else ["Source model", "tgt"]
            )
            + [
                r"\multicolumn{2}{c}{" + metric_label + "}"
                for _ in range(len(train_test_datasets))
                for metric_label in speech_metrics.keys()
            ]
        )
        + r"\\"
    )
    table_content_per_line = [
        before_begin_tabular,
        begin_tabular,
        table_header_datasets,
        table_header_metrics,
    ]
    for model_label, model_results in speech_results_dict[
        supervision_type_label
    ].items():
        table_content_per_line.append("\hline")
        # First append model_label inside a multirow.
        num_rows = len(
            speech_results_dict[supervision_type_label][model_label]
        )
        for i_supervision_variant, (
            supervision_variant_label,
            supervision_variant_results,
        ) in enumerate(model_results.items()):
            if supervision_type_label != "unsupervised":
                if i_supervision_variant == 0:
                    current_line = (
                        r"\multirow{"
                        + str(num_rows)
                        + r"}{*}{"
                        + model_label
                        + "}"
                    )
                else:
                    current_line = ""
                current_line += "& " + supervision_variant_label
            else:
                current_line = supervision_variant_label
            for (
                dataset_label,
                dataset_results,
            ) in supervision_variant_results.items():
                for metric_label, metric_result in dataset_results.items():
                    current_line += " & "
                    if metric_result is not None:
                        if (
                            wpe_results_dict
                            and comparison_with_baseline[
                                supervision_type_label
                            ][model_label][supervision_variant_label][
                                dataset_label
                            ][
                                metric_label
                            ]
                        ):
                            color = "green"
                        else:
                            color = ""
                        current_line += format_array(
                            metric_result,
                            bold=bold_mask is not None
                            and bold_mask[supervision_type_label][model_label][
                                supervision_variant_label
                            ][dataset_label][metric_label],
                            color=color,
                        )
                    else:
                        current_line += " & "
            current_line += r" \\"
            table_content_per_line.append(current_line)
    # trainingless
    if trainingless_results_dict:
        table_content_per_line.append(r"\hline")
        trainingless_res_line = line_of_results(
            results_dict=trainingless_results_dict,
            line_name="Trainingless",
            supervision_type_label=supervision_type_label,
        )
        table_content_per_line.append(trainingless_res_line)

    # WPE metrics
    if wpe_results_dict:
        table_content_per_line.append(r"\hline")
        wpe_res_line = line_of_results(
            results_dict=wpe_results_dict,
            line_name="WPE",
            supervision_type_label=supervision_type_label,
        )
        table_content_per_line.append(wpe_res_line)

    # Reverberant metrics
    table_content_per_line.append(r"\hline")
    reverberant_results_line = line_of_results(
        results_dict=reverberant_results_dict,
        line_name="Reverberant",
        supervision_type_label=supervision_type_label,
    )
    table_content_per_line.append(reverberant_results_line)

    # End of table
    end_tabular_and_after = (
        r"""
    \end{NiceTabular}
  }
  \label{tab:"""
        + f"{supervision_type_label}"
        + r"""}
\end{table*}"""
    )
    table_content_per_line.append(end_tabular_and_after)
    full_table = "\n".join(table_content_per_line)
    with open("tables.tex", "a") as f:
        print("", file=f)
        print(
            full_table,
            file=f,
        )
    return full_table


def make_reverb_model_plot(reverb_model_results_dict):
    fig, axs = plt.subplots(
        len(reverb_model_metrics),
        len(train_test_datasets),
        sharex="col",
        sharey="row",
        figsize=(15, 10),
    )
    for i_metric_label, metric_label in enumerate(reverb_model_metrics.keys()):
        axs[i_metric_label, 0].set_ylabel(metric_label)
        for i_dataset_label, dataset_label in enumerate(
            train_test_datasets.keys()
        ):
            if i_metric_label == 0:
                axs[i_metric_label, i_dataset_label].set_title(dataset_label)
            ax = axs[i_metric_label, i_dataset_label]
            # we flatten reverb_model_dict for that dataset
            data_dict = {}
            for (
                reverb_model_supervision_label
            ) in reverb_model_supervisions.keys():
                for (
                    reverb_model_fraction_label
                ) in reverb_model_fractions.keys():
                    supervision_and_fraction_label = f"{reverb_model_supervision_label} ({reverb_model_fraction_label})"
                    data_dict[supervision_and_fraction_label] = (
                        reverb_model_results_dict[
                            reverb_model_supervision_label
                        ][reverb_model_fraction_label][dataset_label][
                            metric_label
                        ]
                    )
            data_array = np.stack(list(data_dict.values())).T
            ax.violinplot(data_array, showmedians=True)
            # ax.set_yscale("log")
            ax.set_xticks(
                list(range(len(data_dict))),
                # labels=data_dict.keys(),
                labels=[k.replace("percent", "%") for k in data_dict.keys()],
                rotation=45,
            )
            ax.set_xlabel("Training loss (dataset size)")
    fig.suptitle("Reverberation model results")
    fig.savefig(
        os.path.join(save_dir, "reverb_model_results.pdf"), bbox_inches="tight"
    )


def plot_dry_vs_exact_rir_violin(speech_results_dict):
    results_dict_reordered = move_level_to_last(
        move_level_to_last(speech_results_dict, 1), 1
    )["strong"]
    fig, axs = plt.subplots(
        len(speech_metrics),
        len(train_test_datasets),
        sharex=True,
        sharey="row",
    )
    for i_dataset, (dataset_key, dataset_results) in enumerate(
        results_dict_reordered.items()
    ):
        for i_metric, (metric_key, metric_results) in enumerate(
            dataset_results.items()
        ):
            ax = axs[i_metric, i_dataset]
            if i_dataset == 0:
                ax.set_ylabel(metric_key)
            data_array = np.stack(
                [a["RIR"] - a["dry"] for a in metric_results.values()]
            ).T
            positions = list(range(len(metric_results.keys())))
            # ax.violinplot(data_array, showmeans=True, positions=positions)
            ax.boxplot(data_array, positions=positions, showfliers=True)
            ax.hlines(0, xmin=min(positions), xmax=max(positions))
            ax.set_xticks(
                positions,
                labels=metric_results.keys(),
            )


def plot_dry_vs_exact_rir_bar(speech_results_dict):
    reordered_results_dict = move_last_level_to_position(
        move_last_level_to_position(speech_results_dict["strong"], 0), 1
    )
    fig, axs = plt.subplots(
        len(speech_metrics),
        1,
        sharex=True,
        sharey="row",
    )
    width = 1 / (len(speech_models) + 1)
    x = np.arange(len(train_test_datasets))
    for i_metric, (metric_key, results_per_metric) in enumerate(
        reordered_results_dict.items()
    ):
        ax = axs[i_metric]
        ax.set_ylabel(delta_label_for_plot(metric_key))
        datasets = list(results_per_metric.keys())
        models = list(
            next(iter(results_per_metric.values())).keys()
        )  # Assume all datasets have the same models
        model_results = {
            model: [
                (
                    results_per_metric[ds][model]["RIR"]
                    - results_per_metric[ds][model]["dry"]
                ).mean()
                for ds in datasets
            ]
            for model in models
        }
        x = np.arange(len(datasets))
        width = 0.25  # The width of the bars
        for i, model in enumerate(models):
            offset = i - (len(models) - 1) / 2
            ax.bar(
                x + offset * width, model_results[model], width, label=model
            )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.axhline(color="black")
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend()
    fig.legend(
        handles,
        labels,
    )
    fig.savefig(
        os.path.join(save_dir, "results_strong.pdf"), bbox_inches="tight"
    )


def plot_strong_supervision_results_old(speech_results_dict):
    df = dict_to_dataframe(
        speech_results_dict["strong"],
        column_names=["Model", "supervision", "dataset", "metric"],
        wide_metrics=False,
    )
    df_wide = df.pivot(
        columns="supervision",
        index=["Model", "dataset", "metric", "idx_in_dataset"],
        values="value",
    )
    df_wide.reset_index(inplace=True)
    df_wide["diff"] = df_wide["RIR"] - df_wide["dry"]
    data = df_wide.query("metric != 'SRMR'")
    data = df_wide
    g = sns.catplot(
        data=data,
        x="dataset",
        y="diff",
        hue="Model",
        row="metric",
        kind="bar",
        sharey=False,
        row_order=list(speech_metrics.keys())[: len(data["metric"].unique())],
        order=list(train_test_datasets.keys()),
        height=2,
        aspect=5 / 2,
        legend_out=True,
    )
    for metric, ax in g.axes_dict.items():
        ax.set_ylabel(delta_label_for_plot(metric))
        ax.set_title("")
    plt.savefig(
        os.path.join(save_dir, "results_strong.pdf"), bbox_inches="tight"
    )


def plot_strong_supervision_results(
    speech_results_dict, reverberant_results_dict
):
    difference_results_dict = nested_diff(
        speech_results_dict, reverberant_results_dict
    )
    data = dict_to_dataframe(
        difference_results_dict["strong"],
        column_names=["Model", "Supervision", "Dataset", "metric"],
        wide_metrics=False,
    )
    # breakpoint()
    g = sns.catplot(
        data=data,
        x="Model",
        y="value",
        hue="Supervision",
        row="metric",
        col="Dataset",
        kind="bar",
        sharey="row",
        sharex=True,
        row_order=list(speech_metrics.keys())[: len(data["metric"].unique())],
        order=list(speech_models.keys()),
        height=2,
        aspect=0.9,
        # legend_out=True,
    )
    for (metric, dataset), ax in g.axes_dict.items():
        ax.set_ylabel(delta_label_for_plot(metric))
        if metric == next(iter(speech_metrics.keys())):
            ax.set_title(dataset)
        else:
            ax.set_title("")
    g._legend.remove()
    g.add_legend(
        title="Supervision",
        ncols=2,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(
            0.45,
            # 0
            -0.05,
        ),
        adjust_subtitles=True,
    )
    plt.savefig(
        os.path.join(save_dir, "results_strong.pdf"), bbox_inches="tight"
    )


def plot_paired_results(reverberant_results, each_variant_results):
    sort_order = reverberant_results.argsort()
    x = reverberant_results[sort_order]
    results_stacked_sorted = np.stack(list(each_variant_results))[
        :, sort_order
    ]
    plt.plot(x, results_stacked_sorted, ".")


def compare_weak_variants_on_real_rirs(
    speech_results_dict, results_to_compare_to
):
    dataset_above_wpe = list(train_test_datasets.keys())[-1]
    models_above_wpe = list(speech_models.keys())[:2]

    fig, axs = plt.subplots(len(speech_metrics), sharex=True)

    position = np.arange(len(models_above_wpe))

    reordered_results_dict = move_level_to_last(
        move_last_level_to_position(
            move_last_level_to_position(speech_results_dict["weak"], 0), 0
        )[dataset_above_wpe],
        0,
    )
    df = dict_to_dataframe(
        reordered_results_dict,
        column_names=["Model", "variant", "metric"],
    )

    # df.query("model != 'TFL'", inplace=True)
    df_baseline = pd.DataFrame.from_dict(
        results_to_compare_to[dataset_above_wpe]
    )
    df_baseline.index.name = "idx_in_dataset"
    df_joint = df.join(
        df_baseline,
        on="idx_in_dataset",
        how="left",
        rsuffix="_baseline",
        lsuffix="_ours",
        validate="many_to_one",
    )
    for metric_key in speech_metrics:
        df_joint[metric_key] = (
            df_joint[metric_key + "_ours"] - df_joint[metric_key + "_baseline"]
        )
        # df_diff = df_joint.reset_index(inplace=True)

    for i_metric, metric_key in enumerate(speech_metrics.keys()):
        ax = axs[i_metric]
        # df = dict_to_dataframe(
        #     reordered_results_dict[metric_key],
        #     column_names=["Model", "variant"],
        # )
        df_diff = df_joint[["Model", "variant", metric_key]]
        ax.axhline(0, color="lightgray")
        g = sns.violinplot(
            df_diff,
            x="Model",
            y=metric_key,
            hue="variant",
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        # ax.get_legend().remove()
        ax.get_legend().set_visible(False)
        # ax.grid(which="major", axis="y")
        ax.set_ylabel(delta_label_for_plot(metric_key))
        if i_metric != len(speech_metrics) - 1:
            ax.set_xlabel("")
    axs[0].set_ylim(-10, 15)
    axs[1].set_ylim(-0.05, 0.2)
    axs[2].set_ylim(-0.1, 0.5)
    axs[3].set_ylim(-1, 3)
    fig.legend(handles, labels, title="Loss variant")
    fig.savefig(
        os.path.join(save_dir, "weak_variants_comparison_on_ears_reverb.pdf"),
        dpi=600,
        bbox_inches="tight",
    )


def compare_weak_models_regardless_of_variant_old(
    speech_results_dict,
    reverberant_results_dict,
    variant_to_select="single",
):
    df = dict_to_dataframe(
        speech_results_dict["weak"],
        column_names=["Model", "variant", "dataset", "metric"],
        wide_metrics=True,
    )
    df_baseline = dict_to_dataframe(
        reverberant_results_dict,
        column_names=["dataset", "metric"],
        wide_metrics=True,
    )
    df_baseline.set_index(["dataset", "idx_in_dataset"], inplace=True)

    df_joint = df.join(
        df_baseline,
        on=["dataset", "idx_in_dataset"],
        how="left",
        rsuffix="_baseline",
        lsuffix="_ours",
        validate="many_to_one",
    )
    for metric_key in speech_metrics.keys():
        df_joint[metric_key] = (
            df_joint[metric_key + "_ours"] - df_joint[metric_key + "_baseline"]
        )
    df_diff = df_joint[
        [
            cn
            for cn in df_joint.columns
            if "_baseline" not in cn and "_ours" not in cn
        ]
    ]
    df_diff_long = df_diff.melt(
        id_vars=["Model", "variant", "dataset", "idx_in_dataset"],
        var_name="metric",
        value_name="value",
    )
    # sns.barplot(
    #     df_diff,
    #     x="dataset",
    #     y="SISDR",
    #     hue="Model",
    #     col="metric"
    #     order=list(train_test_datasets.keys()),
    # )
    mask_of_variant_to_select = (
        df_diff_long["variant"].str.lower() == variant_to_select
    )
    if mask_of_variant_to_select.any():
        data = df_diff_long[mask_of_variant_to_select]
    else:
        data = df_diff_long
    g = sns.catplot(
        data=df_diff_long,
        kind="bar",
        x="dataset",
        y="value",
        hue="Model",
        row="metric",
        order=list(train_test_datasets.keys()),
        sharey=False,
        height=2,
        aspect=5 / 2,
    )
    for metric, ax in g.axes_dict.items():
        ax.set_ylabel(delta_label_for_plot(metric))
        ax.set_title("")

    g.figure.savefig(
        os.path.join(save_dir, "weak_models_comparison.pdf"),
        bbox_inches="tight",
    )


def compare_weak_models_regardless_of_variant(
    speech_results_dict,
    reverberant_results_dict,
    wpe_results_dict,
    dnn_wpe_results_dict,
    variant_to_select="single",
):
    diff_ours = nested_diff(speech_results_dict, reverberant_results_dict)
    diff_wpe = nested_diff(wpe_results_dict, reverberant_results_dict)
    diff_dnn_wpe = nested_diff(
        dnn_wpe_results_dict[list(reverb_model_fractions.keys())[-1]],
        reverberant_results_dict,
    )

    df_dnn_wpe = dict_to_dataframe(
        diff_dnn_wpe, ["dataset", "metric"], wide_metrics=False
    )
    df_dnn_wpe.query("dataset == 'Real RIRs'", inplace=True)
    df_wpe = dict_to_dataframe(
        diff_wpe, ["dataset", "metric"], wide_metrics=False
    )
    df_wpe["Model"] = "WPE"
    df_dnn_wpe["Model"] = "DNN-WPE"
    df_weak = dict_to_dataframe(
        diff_ours["weak"],
        ["Model", "variant", "dataset", "metric"],
        wide_metrics=False,
    )
    df_weak.query("variant == 'Single'", inplace=True)
    df_weak.drop(columns=["variant"], inplace=True)

    df_full = pd.concat(
        [
            df_weak,
            df_wpe,
            # df_dnn_wpe,
        ],
        ignore_index=True,
    )

    g = sns.catplot(
        data=df_full,
        kind="bar",
        col="dataset",
        y="value",
        hue="Model",
        row="metric",
        sharey="row",
        sharex=True,
        row_order=list(speech_metrics.keys())[
            : len(df_full["metric"].unique())
        ],
        # order=list(speech_models.keys()),
        height=1.8,
        aspect=1,
    )
    for (metric, dataset), ax in g.axes_dict.items():
        ax.set_ylabel(delta_label_for_plot(metric))
        if metric == next(iter(speech_metrics.keys())):
            ax.set_title(dataset)
        else:
            ax.set_title("")
    g._legend.remove()
    g.add_legend(
        title="Model",
        ncols=4,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(
            0.35,
            # -0.02, # if model commented
            -0.05,
        ),
        adjust_subtitles=True,
    )
    # for metric, ax in g.axes_dict.items():
    #     ax.set_ylabel(delta_label_for_plot(metric))
    #     ax.set_title("")

    g.figure.savefig(
        os.path.join(save_dir, "weak_models_comparison.pdf"),
        bbox_inches="tight",
    )
    # Die grosse Bertha
    # g = sns.catplot(
    #     data=df_diff_long,
    #     kind="violin",
    #     x="Model",
    #     y="value",
    #     hue="variant",
    #     col="metric",
    #     row="dataset",
    #     sharey=False,
    # )


def plot_weak_and_strong_supervision(
    speech_results_dict,
    # add_ood_models=["BiLSTM"],
    add_ood_models=[],
):
    strong_results_dict = {
        k: speech_results_dict["strong"][k]["RIR"]
        for k in speech_models.keys()
    }
    weak_results_dict = {
        k: speech_results_dict["weak"][k]["Single"]
        for k in speech_models.keys()
    }
    df_long = dict_to_dataframe(
        {"Strong": strong_results_dict, "Weak": weak_results_dict},
        column_names=["Supervision", "Model", "dataset", "Metric"],
        wide_metrics=False,
    )
    if add_ood_models:
        df_long_strong_ood = dict_to_dataframe(
            move_level_to_last(strong_results_dict, 0)[
                list(train_test_datasets.keys())[1]
            ],
            column_names=["metric", "model"],
            wide_metrics=False,
        )
        df_long_weak_real = dict_to_dataframe(
            move_level_to_last(weak_results_dict, 0)[
                list(train_test_datasets.keys())[2]
            ],
            column_names=["metric", "model"],
            wide_metrics=False,
        )
        df_long_strong_ood.set_index(
            ["metric", "model", "idx_in_dataset"], inplace=True
        )
        df_ood = df_long_weak_real.join(
            df_long_strong_ood,
            on=["metric", "model", "idx_in_dataset"],
            how="left",
            validate="one_to_one",
            lsuffix="_weak_real",
            rsuffix="_strong_ood",
        )
        df_ood["diff"] = df_ood["value_weak_real"] - df_ood["value_strong_ood"]
        df_ood_wide = df_ood.pivot(columns="")

    df_wide = df_long.pivot(
        columns="Supervision",
        index=["Model", "dataset", "Metric", "idx_in_dataset"],
        values="value",
    )
    df_wide.reset_index(inplace=True)
    df_wide["Improvement"] = df_wide["Weak"] - df_wide["Strong"]
    # data = df_wide.query(f"dataset=='{list(train_test_datasets.keys())[-1]}'")
    g = sns.relplot(
        df_wide,
        x="Strong",
        y="Weak",
        hue="Model",
        row="Metric",
        facet_kws=dict(
            sharey=False,
            sharex=False,
        ),
    )
    for ax_name, ax in g.axes_dict.items():
        ax.axline((0, 0), slope=1)

    plt.figure()
    g = sns.catplot(
        df_wide,
        x="Model",
        y="Improvement",
        row="Metric",
        hue="dataset",
        kind="bar",
        sharex=False,
        sharey=False,
    )
    df = df_wide.query(f"dataset=='{list(train_test_datasets.keys())[-1]}'")
    df_ood = df_wide.query(
        f"dataset=='{list(train_test_datasets.keys())[-1]}'"
    )
    df_long_improvement = df.pivot_table(
        index=["Model", "Metric"],
        columns="idx_in_dataset",
        values="Improvement",
        sort=False,
        # aggfunc="mean",
    )
    df_long_strong = df.pivot_table(
        index=["Model", "Metric"],
        columns="idx_in_dataset",
        values="Strong",
        sort=False,
        # aggfunc="mean",
    )
    df_improvement_mean = (
        df_long_improvement.mean(axis=1)
        .unstack(sort=False)
        .T.loc[list(speech_metrics.keys())]
    )
    df_strong_mean = (
        df_long_strong.mean(axis=1)
        .unstack(sort=False)
        .T.loc[list(speech_metrics.keys())]
    )
    df_relative_improvement = df_improvement_mean / df_strong_mean.abs()
    df_relative_improvement = df_improvement_mean

    # df_annot = (
    #     df_long.apply(lambda x: f"{x.mean():.2f} pm {x.std():.2f}", axis=1)
    #     .unstack(sort=False)
    #     .T.loc[list(speech_metrics.keys())]
    # )

    df = df_improvement_mean.T
    normalized_df = (df - df.min()) / (df.max() - df.min())
    normalized_df = (df - df.mean()) / df.std()
    data = normalized_df.T
    fig_height = 2.5
    plt.figure(figsize=(fig_height * 5 / 3, fig_height))
    g = sns.heatmap(
        data,
        # annot=True,
        annot=df_improvement_mean,
        fmt=".2f",
        # norm=LogNorm(),
        cbar=False,
    )
    plt.yticks(
        ticks=np.arange(len(data)) + 0.5,
        labels=[delta_label_for_plot(metric_key) for metric_key in data.index],
        rotation=0,
    )
    plt.savefig("figures/comparison_weak_strong.pdf", bbox_inches="tight")


def plot_unsupervised_results(
    speech_results_dict,
    reverberant_results_dict,
    wpe_results_dict,
    dnn_wpe_results_dict,
    trainingless_results_dict=None,
    results_strong_small_sizes=None,
):
    df_unsupervised = dict_to_dataframe(
        speech_results_dict["unsupervised"],
        column_names=["Model", "variant", "dataset", "metric"],
        wide_metrics=False,
    )
    # split column 'variant' in two
    df_unsupervised = pd.concat(
        (
            df_unsupervised,
            df_unsupervised["variant"]
            .str.split(" & ", n=2, expand=True)
            .rename(columns={0: "supervision", 1: "fraction_str"}),
        ),
        axis=1,
    )
    df_unsupervised.query("dataset=='Real RIRs'", inplace=True)

    # Modify fraction
    df_unsupervised["Dataset size (samples)"] = df_unsupervised[
        "fraction_str"
    ].map(fractions_to_samples)
    # bit of cleaning
    df_unsupervised.drop(columns=["variant", "fraction_str"], inplace=True)

    df_reverberant = dict_to_dataframe(
        reverberant_results_dict,
        column_names=["dataset", "metric"],
        wide_metrics=False,
    )

    results_baselines = {
        "WPE (Baseline)": wpe_results_dict,
        "Trainingless (Ours)": trainingless_results_dict,
        # "DNN-WPE (baseline)": dnn_wpe_results_dict,
    }

    results_strong_supervision_100_examples = {
        k: np.load(
            os.path.join(
                "lightning_logs/test/ears_bilstm_strong_100/version_0/latest_results/"
                + v
            )
        )
        for k, v in speech_metrics.items()
    }
    df_strong_100 = dict_to_dataframe(
        results_strong_supervision_100_examples,
        column_names=["metric"],
        wide_metrics=False,
    )
    df_strong_100["Dataset size (samples)"] = 100
    df_strong_100["supervision"] = "BiLSTM"

    df_strong_ood = dict_to_dataframe(
        speech_results_dict["strong"]["BiLSTM"]["dry"][
            list(train_test_datasets.keys())[1]
        ],
        column_names=["metric"],
        wide_metrics=False,
    )
    df_strong_ood["Dataset size (samples)"] = TRAINING_DATASET_TOTAL_SIZE
    df_strong_ood["supervision"] = "Out-of-domain"

    df_dnn_wpe = dict_to_dataframe(
        dnn_wpe_results_dict,
        column_names=["fraction_str", "dataset", "metric"],
        wide_metrics=False,
    )
    df_dnn_wpe.query("dataset=='Real RIRs'", inplace=True)

    # Modify fraction
    df_dnn_wpe["Dataset size (samples)"] = df_dnn_wpe["fraction_str"].map(
        fractions_to_samples
    )
    # bit of cleaning
    df_dnn_wpe.drop(columns=["fraction_str"], inplace=True)
    df_dnn_wpe["supervision"] = "DNN-WPE"

    df_strong_small_sizes = dict_to_dataframe(
        results_strong_small_sizes["BiLSTM"],
        column_names=["fraction_str", "dataset", "metric"],
        wide_metrics=False,
    )
    df_strong_small_sizes["supervision"] = "BiLSTM"
    df_strong_small_sizes["Dataset size (samples)"] = df_strong_small_sizes[
        "fraction_str"
    ].map(fractions_to_samples)
    # bit of cleaning
    df_strong_small_sizes.drop(
        columns=["fraction_str", "dataset"], inplace=True
    )

    df_strong_full = dict_to_dataframe(
        speech_results_dict["strong"]["BiLSTM"]["dry"][
            list(train_test_datasets.keys())[-1]
        ],
        column_names=["metric"],
        wide_metrics=False,
    )
    df_strong_full["Dataset size (samples)"] = TRAINING_DATASET_TOTAL_SIZE
    df_strong_full["supervision"] = "BiLSTM"

    df_strong_references = pd.concat(
        [
            # df_strong_100,
            # df_strong_ood,
            df_strong_small_sizes,
            df_dnn_wpe,
        ],
        ignore_index=True,
    )

    df_unsupervised["Supervision type"] = "Unsupervised"
    df_strong_references["Supervision type"] = "Strong supervision"
    df_strong_references["dataset"] = list(train_test_datasets.keys())[-1]
    df_ours = pd.concat([df_unsupervised, df_strong_references])
    df_baselines = dict_to_dataframe(
        results_baselines,
        column_names=["variant", "dataset", "metric"],
        wide_metrics=False,
    )
    df_reverberant.set_index(
        ["dataset", "idx_in_dataset", "metric"], inplace=True
    )
    df_ours = df_ours.join(
        df_reverberant,
        on=["dataset", "idx_in_dataset", "metric"],
        how="left",
        lsuffix="_ours",
        rsuffix="_reverberant",
        validate="many_to_one",
    )
    df_ours["Improvement"] = (
        df_ours["value_ours"] - df_ours["value_reverberant"]
    )

    df_baselines = df_baselines.join(
        df_reverberant,
        on=["dataset", "idx_in_dataset", "metric"],
        how="left",
        lsuffix="_baselines",
        rsuffix="_reverberant",
        validate="many_to_one",
    )
    df_baselines["Improvement"] = (
        df_baselines["value_baselines"] - df_baselines["value_reverberant"]
    )
    df_baselines = df_baselines.merge(
        pd.Series(
            list(df_ours["Dataset size (samples)"].unique()),
            name="Dataset size (samples)",
        ),
        how="cross",
    )
    df_ours.reset_index(inplace=True)

    data_baselines = df_baselines.query("dataset=='Real RIRs'")
    g = sns.catplot(
        data=df_ours,
        kind="point",
        x="Dataset size (samples)",
        y="Improvement",
        row="metric",
        hue="supervision",
        markers=["o", "o", "v", "v"],
        # palette=[sns.color_palette()[i] for i in [0, 1, 3, 4]],
        sharey=False,
        native_scale=True,
        height=2,
        aspect=3 / 2,
    )

    for metric_key, ax in g.axes_dict.items():
        # for ax in g.axes.ravel():
        query = ax.get_title().replace("= ", "== '") + "'"
        sns.lineplot(
            data=data_baselines.query(query),
            x="Dataset size (samples)",
            y="Improvement",
            ax=ax,
            style="variant",
            dashes=[(2, 2), (1, 1)],
            # color=sns.color_palette()[2],
            color="grey",
        )
        ax.set_ylabel(delta_label_for_plot(metric_key))
        ax.set_title("")
        ax.get_legend().set_visible(False)
    handles, labels = ax.get_legend_handles_labels()

    (p1,) = plt.plot(
        [0], marker="None", linestyle="None", label="dummy-tophead"
    )

    handles = [
        p1,
        p1,
        handles[0],
        handles[1],
        p1,
        p1,
        handles[2],
        handles[3],
        p1,
        p1,
        handles[4],
        handles[5],
    ]
    labels = [
        "Supervised by",
        "a reverberation model",
        "trained using PM Loss",
        "trained using RM Loss",
        "",
        "Strong supervision",
        labels[2],
        labels[3],
        r"$~$",
        "No training",
        labels[4],
        labels[5],
    ]
    g._legend.set_visible(False)
    plt.xscale("log")
    g.add_legend(dict(zip(labels, handles)))
    # sns.move_legend(g, loc="outside lower center")
    # fig = g.figure
    # fig.legend(
    #     handles,
    #     labels,
    #     # loc="lower center",
    #     # bbox_to_anchor=(0.5, 0),
    #     # ncols=3,
    # )
    plt.savefig(
        os.path.join(save_dir, "results_unsupervised.pdf"),
        bbox_inches="tight",
    )


def plot_reverb_model_results(reverb_model_results_dict):
    df_long = dict_to_dataframe(
        reverb_model_results_dict,
        ["Training Loss", "fraction", "dataset", "test loss"],
        wide_metrics=False,
    )
    df_long["Dataset size (samples)"] = df_long["fraction"].map(
        fractions_to_samples
    )
    data = df_long.query(f"dataset=='{list(train_test_datasets.keys())[-1]}'")
    data = df_long
    g = sns.catplot(
        data=data,
        kind="point",
        x="Dataset size (samples)",
        y="value",
        row="test loss",
        col="dataset",
        hue="Training Loss",
        sharey="row",
        native_scale=True,
        height=2,
        # aspect=5 / 2,
    )

    for (metric, dataset), ax in g.axes_dict.items():
        ax.set_xscale("log")
        ax.set_ylabel(metric + r" metric ($\downarrow$)")
        if metric == list(reverb_model_metrics.keys())[0]:
            ax.set_title(dataset)
            ax.set_yscale("log")
        else:
            ax.set_title("")
    plt.savefig(
        os.path.join(save_dir, "results_reverb_model.pdf"), bbox_inches="tight"
    )

    df_wide = dict_to_dataframe(
        reverb_model_results_dict,
        column_names=["Training Loss", "fraction", "dataset", "metric"],
        wide_metrics=True,
    )
    df_wide["Dataset size (samples)"] = df_wide["fraction"].map(
        lambda x: str(fractions_to_samples[x])
    )

    data = df_wide.query(f"dataset=='{list(train_test_datasets.keys())[-1]}'")
    plt.figure()

    ax = sns.scatterplot(
        data,
        x="RM",
        y="PM",
        hue="Dataset size (samples)",
        style="Training Loss",
        # hue_norm=LogNorm(),
    )
    ax.set_xlabel(r"RM metric ($\downarrow$)")
    ax.set_ylabel(r"PM metric ($\downarrow$)")
    ax.set_yscale("log")
    # We split the legend in two parts
    handles, labels = ax.get_legend_handles_labels()
    legend_sizes = ax.legend(handles[:4], labels[:4])
    ax.add_artist(legend_sizes)
    legend_loss = ax.legend(handles[4:], labels[4:])
    plt.savefig(
        os.path.join(save_dir, "scatterplot_pm_rm.pdf"), bbox_inches="tight"
    )

    # Relative improvement of training on the target loss.
    df = dict_to_dataframe(
        reverb_model_results_dict,
        column_names=["Training Loss", "size", "dataset", "metric"],
        wide_metrics=False,
    )
    df_wide_training_loss = df.pivot(
        columns="Training Loss",
        index=["size", "dataset", "metric", "idx_in_dataset"],
        values="value",
    )
    df_wide_training_loss.reset_index(inplace=True)
    assert len(reverb_model_supervisions.keys()) == 2
    first_model_supervision, second_model_supervision = (
        reverb_model_supervisions.keys()
    )
    df_wide_training_loss["same loss than metric"] = (
        df_wide_training_loss.apply(
            lambda row: (
                row[first_model_supervision]
                if row["metric"] == first_model_supervision
                else row[second_model_supervision]
            ),
            axis=1,
        )
    )
    df_wide_training_loss["different loss than metric"] = (
        df_wide_training_loss.apply(
            lambda row: (
                row[first_model_supervision]
                if row["metric"] == second_model_supervision
                else row[second_model_supervision]
            ),
            axis=1,
        )
    )
    df_wide_training_loss["improvement when training over same loss"] = (
        df_wide_training_loss["different loss than metric"]
        - df_wide_training_loss["same loss than metric"]
    )
    # Pas convaincu par le plot
    # sns.catplot(
    #     data=df_wide_training_loss,
    #     kind="bar",
    #     x="size",
    #     y="improvement when training over same loss",
    #     row="metric",
    #     hue="dataset",
    #     sharey=False,
    # )


def plot_wsj_results(wer_results, wsj_results):
    prune_models_full_of_none(wsj_results)
    wsj_means = apply_to_layer(wsj_results, np.median, 2)
    df_wsj_means = pd.DataFrame(wsj_means).T
    df_means = pd.concat([df_wsj_means, wer_results], axis=1)
    df_means.dropna(inplace=True)

    wsj_diffs = {
        model_label: {
            metric_label: metric_data - wsj_results["wet"][metric_label]
            for metric_label, metric_data in model_data.items()
        }
        for model_label, model_data in wsj_results.items()
    }
    wsj_diffs_selected_models = {
        k: v
        for k, v in wsj_diffs.items()
        if k
        in [
            "nara_wpe",
            "bilstm_strong_100_dnn_wpe",
            "bilstm_unsupervised",
            "bilstm_strong_100",
        ]
    }

    data = dict_to_dataframe(
        wsj_diffs_selected_models,
        column_names=["Method", "metric"],
        wide_metrics=False,
    )

    g = sns.catplot(
        data=data,
        kind="bar",
        y="Method",
        x="value",
        col="metric",
        sharex=False,
        native_scale=True,
        height=3,
        aspect=2 / 2.5 * 2 / 3,
        orient="h",
    )

    # for col_1 in ["ESTOI", "SISDR"]:
    #     plt.figure()
    #     plt.title(col_1)
    #     for col_2 in [
    #         "asr-wav2vec2-commonvoice-en",
    #         "asr-crdnn-rnnlm-librispeech",
    #     ]:
    #         plt.scatter(df[col_1], df[col_2], label=col_2)
    #         for i, label in enumerate(df.index):
    #             plt.text(
    #                 df[col_1][i], df[col_2][i], label, fontsize=10, ha="right"
    #             )


def export_tables_to_pdf():
    current = os.getcwd()
    temp = tempfile.mkdtemp()
    shutil.copy2("tables.tex", temp)
    os.chdir(temp)

    tex = r"""\documentclass{article}
\usepackage{graphics}
\usepackage{amsmath,amsfonts}
\usepackage{siunitx}
\usepackage{stfloats}
\usepackage{multirow}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{nicematrix}
\begin{document}
\input{tables.tex}
\end{document}
"""

    with open("document.tex", "w") as f:
        f.write(tex)
    os.system("pdflatex document.tex > /dev/null 2>&1")
    shutil.copy("document.pdf", os.path.join(current, "tables.pdf"))
    shutil.rmtree(temp)
    os.chdir(current)


# %% Bold mask


def depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0


def print_key_order(d):
    if isinstance(d, dict):
        k, v = next(iter(d.items()))
        print(k, end="|")
        print_key_order(v)


def move_level_to_last(d, level_to_move):
    # Magie noire chatgpt
    def recurse(d, path, level):
        if not isinstance(d, dict):
            yield path, d
        else:
            for k, v in d.items():
                yield from recurse(v, path + [k], level + 1)

    flat_entries = list(recurse(d, [], 0))

    # Determine max depth from the first path
    max_depth = len(flat_entries[0][0])

    # Rearranged entries
    result = lambda: defaultdict(result)
    new_dict = result()

    for path, value in flat_entries:
        if len(path) != max_depth:
            raise ValueError("Inconsistent nesting in dictionary.")

        moved_key = path[level_to_move]
        new_path = (
            path[:level_to_move] + path[level_to_move + 1 :] + [moved_key]
        )

        # Insert value into the new path
        ref = new_dict
        for key in new_path[:-1]:
            ref = ref[key]
        ref[new_path[-1]] = value

    # Convert defaultdicts to dicts recursively
    def to_dict(d):
        if isinstance(d, defaultdict):
            return {k: to_dict(v) for k, v in d.items()}
        return d

    return to_dict(new_dict)


def move_last_level_to_position(d, new_position):
    # Magie noire chatgpt
    def recurse(d, path):
        if not isinstance(d, dict):
            yield path, d
        else:
            for k, v in d.items():
                yield from recurse(v, path + [k])

    flat_entries = list(recurse(d, []))

    # Determine max depth from the first path
    max_depth = len(flat_entries[0][0])

    result = lambda: defaultdict(result)
    new_dict = result()

    for path, value in flat_entries:
        if len(path) != max_depth:
            raise ValueError("Inconsistent nesting in dictionary.")

        moved_key = path[-1]
        new_path = path[:-1]
        new_path.insert(new_position, moved_key)

        ref = new_dict
        for key in new_path[:-1]:
            ref = ref[key]
        ref[new_path[-1]] = value

    # Convert defaultdicts to dicts recursively
    def to_dict(d):
        if isinstance(d, defaultdict):
            return {k: to_dict(v) for k, v in d.items()}
        return d

    return to_dict(new_dict)


def to_dict_of_false(d):
    if isinstance(d, dict):
        return {k: to_dict_of_false(v) for k, v in d.items()}
    return False


def apply_to_layer(d, func, layer):
    if layer == 0:
        return func(d)
    return {k: apply_to_layer(v, func, layer - 1) for k, v in d.items()}


def set_to_true_key(dict_of_bool_to_set, key_to_set_to_true):
    if isinstance(key_to_set_to_true, str | None):
        if (
            key_to_set_to_true is not None
            and key_to_set_to_true.lower() != "none"
        ):
            dict_of_bool_to_set[key_to_set_to_true] = True
        return dict_of_bool_to_set
    return {
        k: set_to_true_key(dict_of_bool_to_set[k], key_to_set_to_true[k])
        for k in key_to_set_to_true.keys()
    }


def key_of_best_mean(d):
    d_mean = {k: v.mean() if v is not None else -np.inf for k, v in d.items()}
    return max(d_mean, key=d_mean.get)


def key_of_best_mean_with_hypothesis_test(d, alpha=0.001):
    best_mean_k = key_of_best_mean(d)
    # we exclude None keys
    other_keys = [
        k for k, v in d.items() if v is not None and k != best_mean_k
    ]
    if len(other_keys) == 1:
        # simple wilcoxon
        pvalue = scipy.stats.wilcoxon(
            d[best_mean_k], d[other_keys[0]], alternative="greater"
        ).pvalue
        if pvalue < alpha:
            # print(best_mean_k)
            return best_mean_k
    elif len(other_keys) > 1:
        # we check if they all follow the same distribution. NOT USED ANYMORE
        pvalue = scipy.stats.friedmanchisquare(
            d[best_mean_k], *(d[k] for k in other_keys)
        ).pvalue
        # if null hypothesis rejected
        if pvalue < alpha or True:
            # compute all other pvalues
            noncorrected_pvalues = [
                scipy.stats.wilcoxon(
                    d[best_mean_k], d[k], alternative="greater"
                ).pvalue
                for k in other_keys
            ]
            # Bonferroni correction
            corrected_pvalues = [
                pvalue * len(noncorrected_pvalues)
                for pvalue in noncorrected_pvalues
            ]
            all_corrected_pvalues_lt_alpha = all(
                cpvalue < alpha for cpvalue in corrected_pvalues
            )
            if all_corrected_pvalues_lt_alpha:
                return best_mean_k
    print(best_mean_k, "non-significant")
    return None

    # return best_mean_k


def compute_bold_mask(
    speech_results_dict,
    supervision_variant_level=2,
    function_to_compute_best=key_of_best_mean_with_hypothesis_test,
):
    # same dict but supervision_variant nested level is last
    speech_results_dict_supervision_variant_last = move_level_to_last(
        speech_results_dict, level_to_move=supervision_variant_level
    )
    best_key_dict_supervision_variant_last = apply_to_layer(
        speech_results_dict_supervision_variant_last,
        function_to_compute_best,
        depth(speech_results_dict_supervision_variant_last) - 1,
    )
    bold_mask_supervision_variant_last = to_dict_of_false(
        speech_results_dict_supervision_variant_last
    )
    bold_mask_supervision_variant_last = set_to_true_key(
        bold_mask_supervision_variant_last,
        best_key_dict_supervision_variant_last,
    )
    bold_mask = move_last_level_to_position(
        bold_mask_supervision_variant_last, supervision_variant_level
    )
    return bold_mask


# %% Some hypothesis tests


def compare_models_for_all_variants(speech_results_dict):
    # move metric to
    reordered_results_dict = move_last_level_to_position(
        speech_results_dict, 1
    )
    reordered_results_dict = move_last_level_to_position(
        reordered_results_dict, 1
    )

    def aux(results_dict, alpha=0.001):
        # I have a dict of models, and for each model a few variants (the same for each model). I want to rank the models. Model A is considered superior to model B if, all variants of model A, perform better than any of the variants for model B. Use scipy's wilcoxon signed-rank test to asses that the results are significant. Results are stored in a depth-2 python dict, where depth 1 is indexed by the model and depth 2 is indexed by the variant and contains a numpy array of the invidual results on a dataset. Create a python function that does this.
        """
        Ranks models based on pairwise statistical comparisons of their variants' performance using the Wilcoxon signed-rank test.

        A model A is considered superior to model B if **all** its variants perform **significantly better** than **all** variants of B.

        Parameters:
            results_dict (dict): Nested dictionary of the form {model: {variant: np.array}}.
            alpha (float): Significance level for the Wilcoxon test.

        Returns:
            list: A ranked list of models from best to worst.
        """

        models = list(results_dict.keys())
        wins = {model: 0 for model in models}

        for model_a, model_b in itertools.combinations(models, 2):
            a_better = True
            b_better = True

            for variant_a in results_dict[model_a]:
                for variant_b in results_dict[model_b]:
                    try:
                        stat, p = scipy.stats.wilcoxon(
                            results_dict[model_a][variant_a],
                            results_dict[model_b][variant_b],
                        )
                    except ValueError:
                        # skip if input arrays are not the same length or have too few elements
                        continue

                    if any(
                        (
                            results_dict[model_a][variant_a] is None,
                            results_dict[model_a][variant_b] is None,
                            results_dict[model_b][variant_a] is None,
                            results_dict[model_b][variant_b] is None,
                        )
                    ):
                        continue
                    # If A is not significantly better than B
                    if not (
                        p < alpha
                        and np.mean(results_dict[model_a][variant_a])
                        > np.mean(results_dict[model_b][variant_b])
                    ):
                        a_better = False
                    # If B is not significantly better than A
                    if not (
                        p < alpha
                        and np.mean(results_dict[model_b][variant_b])
                        > np.mean(results_dict[model_a][variant_a])
                    ):
                        b_better = False

            if a_better and not b_better:
                wins[model_a] += 1
            elif b_better and not a_better:
                wins[model_b] += 1
            # If neither dominates the other, no win is added

        # Rank models by number of wins
        ranked_models = sorted(wins, key=lambda x: wins[x], reverse=True)
        return {k: wins[k] for k in ranked_models}
        return ranked_models

    res = apply_to_layer(
        reordered_results_dict, aux, depth(speech_results_dict) - 2
    )
    return res


def compare_models_with_baseline(
    speech_results_dict, baseline_results_dict, alpha=0.001
):
    # we are at the lowest level
    if speech_results_dict is None:
        return False
    elif isinstance(speech_results_dict, np.ndarray):
        stat, p = scipy.stats.wilcoxon(
            speech_results_dict,
            baseline_results_dict,
        )
        if (
            speech_results_dict.mean() > baseline_results_dict.mean()
            and p < alpha
        ):
            return True
        return False
    # if we are at one of the common levels
    elif speech_results_dict.keys() == baseline_results_dict.keys():

        # we go deeper in both speech_results and baseline results
        return {
            k: compare_models_with_baseline(
                speech_results_dict[k], baseline_results_dict[k]
            )
            for k in speech_results_dict.keys()
        }
    else:
        return {
            k: compare_models_with_baseline(v, baseline_results_dict)
            for k, v in speech_results_dict.items()
        }


def compare_models_with_reverberant(): ...


def is_bilstm_really_better_on_weak_than_strong(speech_results_dict):
    for metric in speech_metrics:
        stat, p = scipy.stats.wilcoxon(
            speech_results_dict["weak"]["BiLSTM"]["Single"]["Real RIRs"][
                metric
            ],
            speech_results_dict["strong"]["BiLSTM"]["RIR"]["Real RIRs"][
                metric
            ],
            alternative="greater",
        )
        print(metric, p)


def compare_models_strong_to_weak(speech_results):
    for metric in speech_metrics:
        for model_1, model_2 in [("BiLSTM", "FSN"), ("FSN", "TFL")]:
            diff_model_1 = (
                speech_results_dict["weak"][model_1]["Single"]["Real RIRs"][
                    metric
                ]
                - speech_results_dict["strong"][model_1]["RIR"]["Real RIRs"][
                    metric
                ]
            )
            diff_model_2 = (
                speech_results_dict["weak"][model_2]["Single"]["Real RIRs"][
                    metric
                ]
                - speech_results_dict["strong"][model_2]["RIR"]["Real RIRs"][
                    metric
                ]
            )
            stat, p = scipy.stats.wilcoxon(
                diff_model_1,
                diff_model_2,
                alternative="greater",
            )
            print(f"{metric}: {model_1} > {model_2}: {p<1e-4}")


def compare_ours_with_dnn_wpe(speech_results_dict, dnn_wpe_results_dict):
    for metric_label in speech_metrics:
        ours = speech_results_dict["unsupervised"]["BiLSTM"][
            "ReverbPM & 100 examples"
        ]["Real RIRs"][metric_label]
        dnn_wpe = dnn_wpe_results_dict["Real RIRs"][metric_label]
        sign = "<" if ours.mean() < dnn_wpe.mean() else ">"
        test = (
            scipy.stats.wilcoxon(ours, dnn_wpe, alternative="greater").pvalue
            < 1e-3
        )
        print(f"{metric_label}: ours {sign} WPE (Significant: {test})")


def plot_results_wrt_rt60(
    metric,
    csv_path="/home/louis/joint-speech-and-reverb-models/data/speech/EARS-Reverb/test_unique_rirs/pairs.csv",
):
    df = pd.read_csv(csv_path)
    rt_60s = list(df["rt_60"])

    y = speech_results_dict["unsupervised"]["BiLSTM"][
        "ReverbPM & 100 examples"
    ]["Real RIRs"][metric]
    y2 = reverberant_results_dict["Real RIRs"][metric]
    plt.figure()
    plt.title(metric)
    plt.scatter(rt_60s, y - y2)
    plt.xlabel(r"$\text{RT}_{60}$")
    plt.ylabel(metric)
    corrcoef = np.corrcoef(rt_60s, y - y2)
    print(f"{metric}: {corrcoef}")


def compare_unsupervised_with_strong_supervision_small_sizes():
    for reverb_model_fraction in reverb_model_fractions:
        for metric_label in speech_metrics:
            ours = speech_results_dict["unsupervised"]["BiLSTM"][
                f"ReverbPM & {reverb_model_fraction}"
            ]["Real RIRs"][metric_label]
            baseline = results_strong_small_sizes["BiLSTM"][
                reverb_model_fraction
            ]["Real RIRs"][metric_label]
            a, b = ours, baseline
            if ours.mean() < baseline.mean():
                sign = "<"
            else:
                sign = ">"
                # raise NotImplementedError()
                # a,b=baseline,ours
                a, b = ours, baseline
            test = (
                scipy.stats.wilcoxon(a, b, alternative="greater").pvalue < 1e-3
            )
            print(
                f"{reverb_model_fraction},{metric_label}: ours {sign} baseline (Significant: {test})"
            )


def compare_phaseinv_with_original_weak(
    speech_results_dict, phase_inv_weak_results
):
    for model_key, model_vals in speech_results_dict["weak"].items():
        if "bilstm" in model_key.lower():
            continue
        for metric_key in speech_metrics.keys():
            a = model_vals["Single"]["Real RIRs"][metric_key]
            b = phase_inv_weak_results[model_key]["phaseinv"]["Real RIRs"][
                metric_key
            ]
            # if a.mean() < b.mean():
            print(
                f"Improvement of phaseinv {model_key}, {metric_key}: {(b-a).mean()}"
            )

    # dict_diff=nested_diff({m:v["Single"] for m,v in speech_results_dict["weak"].items() if "bilstm" not in m.lower()}, phase_inv_weak_results)


# %% Ptflops


def test_models_macs():
    # sshfs store:joint-speech-and-reverb-models/lightning_logs/taslp/reverb_model /home/louis/joint-speech-and-reverb-models/lightning_logs/taslp/reverb_model/

    from ptflops import get_model_complexity_info
    import torch
    from model.utils.run_management import instantiate_model_only

    from types import SimpleNamespace
    from lightning.pytorch.trainer.states import RunningStage
    from torch.utils.flop_counter import FlopCounterMode

    trainer = SimpleNamespace(
        state=SimpleNamespace(stage=RunningStage.TRAINING),
        num_training_batches=1,
        barebones=True,
    )

    config_path = "lightning_logs/taslp/unsupervised/ears_bilstm_null_100/version_1/config.yaml"

    class TrainingLoopWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model._current_fx_name = "toto"
            self.model.speech_model.trainer = trainer
            if self.model.reverb_model is not None:
                self.model.reverb_model.trainer = trainer
            if self.model.joint_loss_module is not None:
                self.model.joint_loss_module.trainer = trainer

        def forward(self, s):
            h = torch.randn(size=(1, 1, 48000))
            y = torch.concat((s, h))
            batch = (y, (s, h, dict()))
            return self.model.training_step(batch, batch_idx=0)

    # Not reporte
    # class JointLossModuleWrapper(TrainingLoopWrapper):
    #     def forward(self, s):
    #         S=self.model.speech_model.stft_module

    with torch.cuda.device(0):
        model_to_evaluate = instantiate_model_only(
            config_path,
            # remove_reverb_model_and_joint_loss=False,
            remove_reverb_model_and_joint_loss=True,
        ).eval()

        model_to_evaluate.speech_model.metrics = torch.nn.ModuleList([])
        full_net = TrainingLoopWrapper(model_to_evaluate)

        # dereverberation_only_net=DereverberationWrapper(model_to_evaluate)

        flop_counter = FlopCounterMode(mods=full_net, display=True, depth=None)
        with flop_counter:
            full_net.forward(torch.rand((1, 1, 48000)))
            total_flops = flop_counter.get_total_flops()
        macs, params = get_model_complexity_info(
            full_net, (1, 48000), backend="aten", print_per_layer_stat=True
        )
    print(total_flops)
    # print(flop_counter.get_table())
    print(macs, params)


# %% Main


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"""\usepackage{amsmath}""",
            # "axes.titlesize":"medium",
            "font.size": 10,
            "font.family": "serif",
            "lines.linewidth": 1,
        }
    )
    plt.close("all")
    os.makedirs(save_dir, exist_ok=True)
    speech_results_dict = gather_results_speech_model()
    wpe_results_dict = gather_results_best_variant_wpe(
        force_best_variant="nara_3_3_10"
    )
    dnn_wpe_results_dict = dict(
        zip(
            reverb_model_fractions.keys(),
            (
                gather_results_best_variant_wpe(
                    force_best_variant="dnn_wpe_bilstm_strong_100_3_3_10"
                ),
                gather_results_best_variant_wpe(
                    force_best_variant="dnn_wpe_bilstm_strong_5percent_3_3_10"
                ),
                gather_results_best_variant_wpe(
                    force_best_variant="dnn_wpe_bilstm_strong_3_3_10"
                ),
            ),
        )
    )

    trainingless_results_dict = gather_results_best_variant_trainingless()
    reverb_model_results_dict = gather_results_reverb_model()
    reverberant_results_dict = gather_reverberant_results()
    prune_models_full_of_none(speech_results_dict)
    wsj_results = gather_results_wsj()
    wer_results = gather_wers()
    results_strong_small_sizes = (
        gather_results_strong_supervision_smaller_sizes()
    )
    phase_inv_weak_results = gather_results_phase_invariant()
    with sns.color_palette("muted"), sns.axes_style("whitegrid"):
        pass
        plot_strong_supervision_results(
            speech_results_dict, reverberant_results_dict
        )
        compare_weak_models_regardless_of_variant(
            speech_results_dict,
            reverberant_results_dict,
            wpe_results_dict=wpe_results_dict,
            dnn_wpe_results_dict=dnn_wpe_results_dict,
        )
        compare_weak_variants_on_real_rirs(
            speech_results_dict,
            results_to_compare_to=reverberant_results_dict,
        )
        plot_reverb_model_results(reverb_model_results_dict)
        plot_unsupervised_results(
            speech_results_dict,
            reverberant_results_dict,
            wpe_results_dict,
            dnn_wpe_results_dict=dnn_wpe_results_dict,
            trainingless_results_dict=trainingless_results_dict,
            results_strong_small_sizes=results_strong_small_sizes,
        )
        # plot_wsj_results(wer_results, wsj_results)
        # for metric in speech_metrics.keys():
        #     plot_results_wrt_rt60(metric)
    #     plot_weak_and_strong_supervision(speech_results_dict)
    # plot_wsj_results(wer_results, wsj_results)
    # test_models_macs()
    # compare_unsupervised_with_strong_supervision_small_sizes()
    # is_bilstm_really_better_on_weak_than_strong(speech_results_dict)
    # compare_models_strong_to_weak(speech_results_dict)
    # make_reverb_model_plot(reverb_model_results_dict=reverb_model_results_dict)
    # res = compare_models_for_all_variants(speech_results_dict)
    # compare_ours_with_dnn_wpe(
    #     speech_results_dict=speech_results_dict,
    #     dnn_wpe_results_dict=dnn_wpe_results_dict_strong_100,
    # )
    # compare_phaseinv_with_original_weak(speech_results_dict, phase_inv_weak_results)
    # bold_mask = compute_bold_mask(speech_results_dict)
    # open("tables.tex", "w").close()
    # full_table = make_table(
    #     speech_results_dict,
    #     wpe_results_dict=dict(),
    #     reverberant_results_dict=reverberant_results_dict,
    #     supervision_type_label="strong",
    #     merge_datasets=True,
    #     bold_mask=bold_mask,
    # )
    # full_table = make_table(
    #     speech_results_dict,
    #     wpe_results_dict=wpe_results_dict,
    #     reverberant_results_dict=reverberant_results_dict,
    #     supervision_type_label="weak",
    #     merge_datasets=True,
    #     bold_mask=bold_mask,
    # )
    # full_table = make_table(
    #     speech_results_dict,
    #     wpe_results_dict=wpe_results_dict,
    #     reverberant_results_dict=reverberant_results_dict,
    #     trainingless_results_dict=trainingless_results_dict,
    #     supervision_type_label="unsupervised",
    #     merge_datasets=True,
    #     bold_mask=bold_mask,
    # )
    # export_tables_to_pdf()
