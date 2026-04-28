import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from collections import Counter
from tabulate import tabulate

BASE_DIR = "lightning_logs/tests/ICASSP2026"
OUTPUT_DIR = "outputs/statistical_tests_icassp2026"

import matplotlib.path as mpath
import matplotlib.patches as mpatches


def add_label_band(
    ax, right, left, label, *, spine_pos=-0.04, tip_pos=-0.01, fontsize=12
):
    """
    Helper function to add bracket around x-tick labels.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the bracket to

    right, left : floats
        The positions in *data* space to bracket on the x-axis

    label : str
        The label to add to the bracket

    spine_pos, tip_pos : float, optional
        The position in *axes fraction* of the spine and tips of the bracket.
        These will typically be negative

    fontsize : int, optional
        The fontsize of the label

    Returns
    -------
    bracket : matplotlib.patches.PathPatch
        The "bracket" Artist.  Modify this Artist to change the color etc of
        the bracket from the defaults.

    txt : matplotlib.text.Text
        The label Artist.  Modify this to change the color etc of the label
        from the defaults.

    """
    # grab the xaxis blended transform
    transform = ax.get_xaxis_transform()

    # add the bracket
    bracket = mpatches.PathPatch(
        mpath.Path(
            [
                [right, tip_pos],
                [right, spine_pos],
                [left, spine_pos],
                [left, tip_pos],
            ]
        ),
        transform=transform,
        clip_on=False,
        facecolor="none",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_artist(bracket)

    # add the label
    txt = ax.text(
        (right + left) / 2,
        spine_pos - 0.03,
        label,
        ha="center",
        va="top",
        rotation="horizontal",
        clip_on=False,
        transform=transform,
        fontsize=fontsize,
    )

    return bracket, txt


def get_latest_versions_path_list(base_dir):
    """
    Get a list of paths to the latest versions of files in the given base directory.
    """
    list_dirs = os.listdir(base_dir)
    latest_versions = []
    for dir_name in list_dirs:
        if os.path.isdir(os.path.join(base_dir, dir_name)):
            dir_path = os.path.join(base_dir, dir_name)
            dir_path = os.path.join(
                dir_path, os.listdir(dir_path)[0]
            )  # Assuming the first subdirectory is the one we want
            if os.path.isdir(dir_path):
                # Get the latest version file in the directory
                version_dir = os.listdir(dir_path)[
                    -1
                ]  # Assuming the last one is the latest
                version_path = os.path.join(dir_path, version_dir, "latest_results")
                latest_versions.append(version_path)
    return latest_versions


def get_metric_files(exp_dir):
    """
    Get a list of test files in the given experiment directory.
    """
    test_files = []
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith(".npy"):
                test_files.append(os.path.join(root, file))
    return test_files


def load_metrics(exp_dir, input_metrics=False):
    """
    Load metrics from a given experiment directory.
    """
    metrics = {}
    for file in get_metric_files(exp_dir):
        cond = (input_metrics and file.endswith("_input.npy")) or (
            not input_metrics and not file.endswith("_input.npy")
        )
        if cond:
            file_name = os.path.basename(file)
            if "test_step" in file_name:
                pass  # The metric is not relevant since it depends on the dereverberator's internal loss, which is not always phase invariant
            elif "Perceptual" in file_name:
                metrics["WB-PESQ"] = np.load(file, allow_pickle=True)
            elif "Scale" in file_name:
                metrics["SISDR"] = np.load(file, allow_pickle=True)
            elif "Short" in file_name:
                metrics["ESTOI"] = np.load(file, allow_pickle=True)
            elif "SRMR" in file_name:
                metrics["SRMR"] = np.load(file, allow_pickle=True)
            else:
                raise ValueError(f"Unknown metric file: {file_name}")
    return metrics


def get_exp_name_from_path(exp_path):
    """
    Extract the experiment name from the given path.
    """
    parts = exp_path.split(os.sep)
    if len(parts) > 0:
        full_name = parts[-4]  # Return the last part of the path as the experiment name
        return full_name.removeprefix("ears16_")
    else:
        raise ValueError("Invalid experiment path provided.")


def violin_plot_metrics(
    metrics_per_experiment, metrics_to_plot, model_idx={}
):  # TODO : automatically order the experiments based on model type, then PI loss usage, then compression usage, and input at the end.
    """
    Create box plots for the specified metrics across different experiments.
    """

    N_metrics = len(metrics_to_plot)
    if N_metrics == 0:
        raise ValueError("No metrics to plot provided.")
    n_cols = min(N_metrics, 4)
    n_rows = (
        N_metrics + n_cols - 1
    ) // n_cols  # Calculate number of rows needed for subplots
    # Maximum of 2 columns

    # Set up the matplotlib figure
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=2.0)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False
    )

    # Sort the experiments based on model_idx and build a dataframe for seaborn
    metrics_per_experiment = dict(
        sorted(metrics_per_experiment.items(), key=lambda x: model_idx[x[0]])
    )
    metrics_per_experiment_df_transposed = pd.DataFrame(metrics_per_experiment).T

    # Define hatches for the legends
    hatches = [
        "///" if exp_name.endswith("compressed") or exp_name == "PI-FSN" else ""
        for exp_name in metrics_per_experiment.keys()
    ]

    # Define a function to extract model type from experiment name
    get_model_type_from_exp_name = lambda exp_name: (
        "FSN" if exp_name.startswith("FSN") else exp_name
    )

    # Create violin plots for each metric
    for k, metric in enumerate(metrics_to_plot):

        # convert each metric collection to a dataframe suitable for seaborn
        metric_df = pd.DataFrame(dict(metrics_per_experiment_df_transposed[metric]))
        metric_df = metric_df.melt(
            value_vars=metric_df.columns, var_name="Experiment", value_name=metric
        )

        # Extract experiment attributes
        metric_df["compression"] = metric_df["Experiment"].apply(
            lambda exp_name: (
                "yes"
                if exp_name.endswith("compressed") or exp_name == "PI-FSN"
                else "None" if exp_name == "input" else "no"
            )
        )
        metric_df["PI loss"] = metric_df["Experiment"].apply(
            lambda exp_name: (
                "yes"
                if "phase_inv" in exp_name
                or "PhaseInv" in exp_name
                or exp_name == "PI-FSN"
                else "None" if exp_name == "input" else "no"
            )
        )

        # get subplot location
        i, j = divmod(k, n_cols)
        ax = axs[i, j]

        # do the violin plot
        sns.violinplot(
            data=metric_df,
            x="Experiment",
            y=metric,
            hue="PI loss",
            palette=["lightblue", "orange", "lightgray"],
            density_norm="width",
            ax=ax,
        )

        # add hatches to indicate if compression was used
        ihatch = iter(hatches)
        for i in ax.get_children():
            if isinstance(i, mpl.collections.PolyCollection):
                i.set_hatch(next(ihatch))
        ax.legend_.remove()  # legend is removed from individual plots to be added globally later
        ax.set_xlabel("")

        # label xaxis with model types and gather repeated model types under one bracket
        model_types = [
            get_model_type_from_exp_name(exp_name)
            for exp_name in metrics_per_experiment.keys()
        ]
        model_types_counter = Counter(model_types)
        ticks = []
        labels = []
        for model_type, count in model_types_counter.items():
            if count > 1:
                first_idx = model_types.index(model_type)
                last_idx = first_idx + count - 1
                add_label_band(
                    ax,
                    right=last_idx + 0.5,
                    left=first_idx - 0.5,
                    label=model_type,
                    fontsize=20,
                )
            else:
                ticks.append(model_types.index(model_type))
                labels.append(model_type)
        ax.set_xticks(
            ticks=ticks,
            labels=labels,
            fontsize=20,
            rotation=30,
            ha="center",
        )

    legend_elements = [
        mpl.patches.Patch(
            facecolor="lightblue", edgecolor="k", label="Phase-dependent loss"
        ),
        mpl.patches.Patch(
            facecolor="orange", edgecolor="k", label="Phase-invariant loss"
        ),
        mpl.patches.Patch(
            facecolor="white", edgecolor="k", label="No compression", hatch=""
        ),
        mpl.patches.Patch(
            facecolor="white", edgecolor="k", label="Log compression", hatch="///"
        ),
    ]

    plt.tight_layout(pad=2.5, w_pad=1.0, h_pad=2.0)
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        fontsize=18,
    )
    return fig, axs


def process_metrics_and_violin_plot(model_name=None, savefig=False, **other_kwargs):
    versions_dir_list = get_latest_versions_path_list(BASE_DIR)
    metrics_per_experiment = {}
    for version_dir in versions_dir_list:
        exp_name = get_exp_name_from_path(version_dir)
        exp_name = convert_exp_name(exp_name)
        if model_name == None or exp_name.startswith(model_name):
            print(f"Processing version directory: {version_dir}")
            metrics = load_metrics(version_dir, input_metrics=False)
            metrics_per_experiment[exp_name] = metrics
    input_metrics = load_metrics(version_dir, input_metrics=True)
    metrics_per_experiment["input"] = input_metrics
    fig, axs = violin_plot_metrics(
        metrics_per_experiment,
        metrics_to_plot=["SRMR", "SISDR", "WB-PESQ", "ESTOI"],
        **other_kwargs,
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if savefig:
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"violin_plots_all_models.svg"),
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()


def compute_metrics_per_rir_partition(exp_dir, input_metrics=False, metrics=None):
    EARS_ROOT = "data/speech/EARS-Reverb_v2"
    pairs = pd.read_csv(os.path.join(EARS_ROOT, "test_unique_rirs/pairs.csv"))
    partitions = json.load(
        open(os.path.join(EARS_ROOT, "rescaled_test_rt60_partitions.json"), "r")
    )
    if metrics is None:
        metrics = load_metrics(exp_dir, input_metrics=input_metrics)
    metrics_per_partition = {
        part: {metric: [] for metric in metrics.keys()} for part in partitions.keys()
    }
    for metric in metrics.keys():
        for part, rir_list in partitions.items():
            # print(part, rir_list)
            for i, rir_idx in enumerate(pairs["rir_idx"]):
                if rir_idx in rir_list:
                    metrics_per_partition[part][metric].append(metrics[metric][i])
            # Convert lists to numpy arrays for easier plotting
            metrics_per_partition[part][metric] = np.array(
                metrics_per_partition[part][metric]
            )
    return metrics_per_partition


def compute_statistics(metrics_per_experiments):
    means_per_experiments, vars_per_experiments = {}, {}
    for exp_name, metrics in metrics_per_experiments.items():
        exp_new_name = convert_exp_name(exp_name)
        means_per_experiments[exp_new_name] = {}
        vars_per_experiments[exp_new_name] = {}
        for metric in metrics:
            means_per_experiments[exp_new_name][metric] = np.mean(
                metrics_per_experiments[exp_name][metric]
            )
            vars_per_experiments[exp_new_name][metric] = np.var(
                metrics_per_experiments[exp_name][metric]
            )
    return means_per_experiments, vars_per_experiments


def convert_exp_name(exp_name):
    if exp_name.startswith("PhaseInvFSN_"):
        return "PI-FSN"
    elif exp_name.startswith("FSN_"):
        if "linear" in exp_name:
            if "compressed" in exp_name:
                return "FSN-PhaseDep-compressed"
            else:
                return "FSN-PhaseDep-vanilla"
        elif "logloss" in exp_name:
            return "FSN-PhaseInv-compressed"
        elif "phase_inv_mse" in exp_name:
            return "FSN-PhaseInv-vanilla"
    return exp_name


def print_statistics(means, vars, latex=False, model_idx={}):
    table = []

    headers = ["Model"] + list(means[list(means.keys())[0]].keys())
    for exp_name in means.keys():
        new_name = exp_name
        table.append([f"{new_name}"])
        dict_of_stats = {}
        for metric in means[exp_name].keys():
            mean, std = means[exp_name][metric], vars[exp_name][metric] ** 0.5
            dict_of_stats[metric] = f"{mean:.3f} ± {std:.2e}"
        list_of_stats = [dict_of_stats[metric] for metric in headers[1:]]
        table[-1] = table[-1] + list_of_stats
    table = sorted(table, key=lambda x: model_idx[x[0]])  # Sort by model name
    if latex:
        print(tabulate(table, headers=headers, tablefmt="latex"))
    else:
        print(tabulate(table, headers=headers, tablefmt="grid"))


def compute_and_print_statistics(latex=False, **kwargs):
    versions_dir_list = get_latest_versions_path_list(BASE_DIR)
    metrics_per_experiment = {}
    for version_dir in versions_dir_list:
        exp_name = get_exp_name_from_path(version_dir)
        metrics = load_metrics(version_dir, input_metrics=False)
        metrics_per_experiment[exp_name] = metrics
    input_metrics = load_metrics(versions_dir_list[1], input_metrics=True)
    metrics_per_experiment["input"] = input_metrics
    means, vars = compute_statistics(metrics_per_experiment)
    print_statistics(means, vars, latex=latex, **kwargs)


if __name__ == "__main__":
    model_idx = {
        "input": 5,
        "FSN-PhaseInv-compressed": 3,
        "FSN-PhaseInv-vanilla": 2,
        "FSN-PhaseDep-compressed": 1,
        "FSN-PhaseDep-vanilla": 0,
        "PI-FSN": 4,
    }
    compute_and_print_statistics(
        latex=False,
        model_idx=model_idx,
    )
    process_metrics_and_violin_plot(model_idx=model_idx, savefig=True)
