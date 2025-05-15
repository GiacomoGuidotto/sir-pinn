import glob
import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns


FIGURES_DIR = "./figures"
LOG_DIR = "./data/logs"

@dataclass
class Variation:
    title: str
    file_name: str
    legend_description: str


@dataclass
class Reference:
    version: str
    legend_description: str


@dataclass
class Metric:
    col: str
    title: str
    log_scale: bool = False
    legend_loc: str = "best"


@dataclass
class PlotConfig:
    name: str
    variations: List[Variation]
    reference: Reference
    metrics: List[Metric]

    @classmethod
    def from_dict(cls, data: Dict) -> "PlotConfig":
        """Create a PlotConfig from a dictionary."""
        return cls(
            name=data["name"],
            variations=[Variation(**v) for v in data["variations"]],
            reference=Reference(**data["reference"]),
            metrics=[Metric(**m) for m in data["metrics"]],
        )


def find_folders(pattern: str) -> List[str]:
    folders = glob.glob(pattern, recursive=True)
    if not folders:
        raise ValueError(f"No folders found matching pattern: {pattern}")
    return folders


def load_dataframe(folder: str) -> Optional[pd.DataFrame]:
    metrics_file = os.path.join(folder, "metrics.csv")
    if not os.path.exists(metrics_file):
        return None

    df = pd.read_csv(metrics_file)
    df["version"] = os.path.basename(folder)

    return df


def load_metrics(
    variation: Variation, reference: Reference
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    pattern = os.path.join(LOG_DIR, "csv", "**", f"*{variation.file_name}*")
    folders = find_folders(pattern)

    dfs = [df for folder in folders if (df := load_dataframe(folder)) is not None]
    if not dfs:
        raise ValueError(f"No metrics found for pattern: {variation.file_name}")

    ref_pattern = os.path.join(LOG_DIR, "csv", "**", f"*{reference.version}_*")
    ref_folders = find_folders(ref_pattern)

    print(
        f"Plot \"{variation.file_name}\", reference {reference.version}: plotting {', '.join(folders)}..."
    )

    ref_df = load_dataframe(ref_folders[0])
    if ref_df is None:
        raise ValueError(f"Reference metrics file not found")

    ref_full_name = os.path.basename(ref_folders[0])
    ref_df["version"] = "reference"

    metrics_df = pd.concat(dfs, ignore_index=True)
    metrics_df = clean_dataframe(metrics_df)
    ref_df = clean_dataframe(ref_df)

    return metrics_df, ref_df, ref_full_name


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe by removing NaN values and ensuring numeric types."""
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    clean_df = df.dropna(subset=["epoch", "train/beta"])
    clean_df = clean_df.sort_values("epoch")

    if len(clean_df) == 0:
        print("WARNING: No valid data remains after cleaning!")

    return clean_df


def plot_metric(
    ax,
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    variation: Variation,
    reference: Reference,
    log_scale: bool = False,
    legend_loc: str = "best",
):
    ref_df_plot = ref_df.dropna(subset=[y_col])
    df_plot = df.dropna(subset=[y_col])

    if not ref_df_plot.empty:
        ax.plot(
            ref_df_plot[x_col],
            ref_df_plot[y_col],
            color="gray",
            linestyle="--",
            label=f"({reference.version}) {reference.legend_description} (reference)",
        )

    for version, group in df_plot.groupby("version"):
        v_num = str(version).split("_")[0] if "_" in str(version) else str(version)
        ax.plot(
            group[x_col],
            group[y_col],
            label=f"({v_num}) {variation.legend_description}",
            alpha=0.7,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)

    if log_scale:
        ax.set_yscale("log")

    ax.legend(loc=legend_loc, framealpha=0.7)


def plot_metrics_grid(
    variations: List[Variation],
    reference: Reference,
    metrics: List[Metric],
    output_name: str,
):
    n_rows = len(variations)
    n_cols = len(metrics)

    fig = plt.figure(layout="constrained", figsize=(7 * n_cols, 4 * n_rows))
    subfigs = fig.subfigures(n_rows, 1, hspace=0.05)

    for i, variation in enumerate(variations):
        subfig = subfigs[i]
        subfig.suptitle(variation.title, fontsize="x-large")
        axs = subfig.subplots(1, n_cols)

        metrics_df, reference_df, _ = load_metrics(variation, reference)

        for j, metric in enumerate(metrics):
            ax = axs[j]
            plot_metric(
                ax,
                metrics_df,
                reference_df,
                "epoch",
                metric.col,
                metric.title,
                variation=variation,
                reference=reference,
                log_scale=metric.log_scale,
                legend_loc=metric.legend_loc,
            )

    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


def load_config_file(config_file: str) -> PlotConfig:
    """Load a JSON configuration file and convert to PlotConfig."""
    with open(config_file, "r") as f:
        config_data = json.load(f)
    return PlotConfig.from_dict(config_data)


def main():
    """Process all JSON config files in the figures directory."""
    sns.set_theme(style="darkgrid")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    config_files = glob.glob(os.path.join(FIGURES_DIR, "*.json"))

    if not config_files:
        print(f"No JSON configuration files found in {FIGURES_DIR}")
        return

    print(f"Found {len(config_files)} configuration files")

    for config_file in config_files:
        print(f"\nProcessing {config_file}...")
        config = load_config_file(config_file)
        plot_metrics_grid(
            variations=config.variations,
            reference=config.reference,
            metrics=config.metrics,
            output_name=config.name,
        )


if __name__ == "__main__":
    main()
