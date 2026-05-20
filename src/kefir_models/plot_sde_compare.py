import argparse
import os
from pathlib import Path

import pandas as pd


def save_comparison_plot(
    comparison_frame: pd.DataFrame,
    trial_columns: list[str],
    interval_level: float,
    path: Path,
) -> None:
    """Save observed data, ODE fit, and SDE predictive interval plot iteratively."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(right=0.65, bottom=0.15)
    time = comparison_frame["time"]

    # Pre-calculate axis limits to fix the layout across the sequence
    all_y_cols = [
        "observed_mean", "classical_logistic_fitted_mean",
        "ode_fitted_mean", "sde_mean", "sde_lower", "sde_upper"
    ]
    for col in trial_columns:
        all_y_cols.append(f"{col}")

    valid_cols = [c for c in all_y_cols if c in comparison_frame.columns]
    y_min = comparison_frame[valid_cols].min().min()
    y_max = comparison_frame[valid_cols].max().max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    axis.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    x_min, x_max = time.min(), time.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    axis.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    colors = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
    markers = ["o", "s", "^", "v", "D", "p", "*", "X"]

    axis.set_title("Water Kefir: Neural ODE vs Neural SDE vs Logistic ODE")
    axis.set_xlabel("Time (hrs)")
    axis.set_ylabel(r"Kefir wet biomass $(\mathrm{g/L})$")
    axis.grid(alpha=0.25)

    def save_step(step_num: int, suffix: str):
        handles, labels = axis.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axis.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")
        step_path = path.with_name(f"{step_num:02d}_{path.stem}_{suffix}{path.suffix}")
        fig.savefig(step_path, dpi=300)
        print(f"Saved step {step_num}: {step_path}")

    # 1. Plot the data
    for index, column in enumerate(trial_columns):
        marker = markers[index % len(markers)]
        color = colors[index % len(colors)]
        axis.plot(
            time,
            comparison_frame[f"{column}_observed"],
            marker=marker,
            linestyle="",
            markersize=10,
            color=color,
            markeredgecolor="black",
            alpha=0.25,
            label=f"{column}",
        )
    save_step(1, "data")

    # 2. Plot the mean of the data
    axis.plot(
        time,
        comparison_frame["observed_mean"],
        color="#111111",
        linewidth=2.5,
        linestyle=":",
        label="Observed mean",
    )
    save_step(2, "obs_mean")

    # 3. Plot the classic fitting
    axis.plot(
        time,
        comparison_frame["classical_logistic_fitted_mean"],
        color="#2a9d8f",
        linewidth=2.5,
        linestyle='dotted',
        label="Logistic-ODE fit",
    )
    save_step(3, "classic_fit")

    # 4. Plot the NODE fitting
    axis.plot(
        time,
        comparison_frame["ode_fitted_mean"],
        color="#2364aa",
        linewidth=2.5,
        linestyle="--",
        label="Neural-ODE fit",
    )
    save_step(4, "node_fit")

    # 5. Plot the estimated mean with SDE
    axis.plot(
        time,
        comparison_frame["sde_mean"],
        color="#d95f02",
        linewidth=2.5,
        linestyle="-.",
        label="Neural-SDE mean",
    )
    save_step(5, "sde_mean")

    # 6. Plot the confidence band
    if "sde_lower" in comparison_frame.columns and "sde_upper" in comparison_frame.columns:
        axis.fill_between(
            time,
            comparison_frame["sde_lower"],
            comparison_frame["sde_upper"],
            color="#2dcdb0b3",
            alpha=0.2,
            label=f"Neural SDE {interval_level:.0%} interval",
        )
        axis.plot(
            time, comparison_frame["sde_lower"],
            color="#2dcdb0b3", linewidth=1.0
        )
        axis.plot(
            time, comparison_frame["sde_upper"],
            color="#2dcdb0b3", linewidth=1.0
        )

    save_step(6, "sde_band")
    plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot comparison trajectories from SDE/ODE fit output.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("neural_sde_comparison_outputs/water_kefir_neural_dynamics_comparison.csv"),
        help="Path to the comparison values CSV file.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("neural_sde_comparison_outputs/water_kefir_neural_dynamics_comparison_replot.png"),
        help="Path where the generated plot will be saved.",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
        help="Interval level used for the label (e.g. 0.90).",
    )
    return parser.parse_args()


def main() -> None:
    """Run plotting from the command line."""
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    comparison_frame = pd.read_csv(args.input_csv)

    # Infer trial columns
    # We look for columns ending with "_observed" and ignore "observed_mean"
    trial_columns = [
        col.replace("_observed", "")
        for col in comparison_frame.columns
        if col.endswith("_observed") and col != "observed_mean"
    ]

    if not trial_columns:
        raise ValueError("Could not find any trial columns in the input CSV.")

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    save_comparison_plot(comparison_frame, trial_columns, args.interval_level, args.output_plot)
    print(f"Plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
