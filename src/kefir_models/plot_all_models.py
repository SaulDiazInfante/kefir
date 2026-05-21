"""Plot a step-by-step comparison across all fitted kefir models.

The sequence combines the saved Neural ODE/SDE comparison outputs with the
saved logistic PINN outputs. It is intentionally a plotting-only script: it
does not retrain or resimulate any model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


COLORS = {
    "observed_mean": "#111111",
    "classical_logistic": "#2a9d8f",
    "neural_ode": "#2364aa",
    "neural_sde": "#e66101",
    "neural_sde_band": "#2dcdb0",
    "deterministic_pinn": "#6a3d9a",
    "stochastic_pinn_drift": "#c51b7d",
    "logistic_pinn_sde": "#8c2d04",
    "logistic_pinn_sde_band": "#fdae61",
}
TRIAL_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
TRIAL_MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X"]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Plot a step-by-step comparison across all kefir models.",
    )
    parser.add_argument(
        "--neural-comparison-csv",
        type=Path,
        default=Path(
            "outputs/neural_sde_comparison_outputs/"
            "water_kefir_neural_dynamics_comparison.csv",
        ),
        help="CSV produced by kefir-sde-compare.",
    )
    parser.add_argument(
        "--pinn-comparison-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/water_kefir_logistic_pinn_comparison.csv"),
        help="Observation-grid CSV produced by kefir-logistic-pinn-compare.",
    )
    parser.add_argument(
        "--pinn-dense-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/water_kefir_logistic_pinn_dense.csv"),
        help="Dense-grid PINN CSV produced by kefir-logistic-pinn-compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/all_model_comparison_outputs"),
        help="Directory where the all-model sequence PNGs are written.",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
        help="Predictive interval level shown in legend labels.",
    )
    return parser.parse_args(args)


def infer_trial_columns(frame: pd.DataFrame) -> list[str]:
    """Return trial base names from columns ending with '_observed'."""

    return [
        column.removesuffix("_observed")
        for column in frame.columns
        if column.endswith("_observed") and column != "observed_mean"
    ]


def _choose_pinn_frame(
    observation_frame: pd.DataFrame,
    dense_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Use dense PINN curves when available, otherwise observation-grid curves."""

    if dense_frame is not None and not dense_frame.empty:
        return dense_frame
    return observation_frame


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column with invalid values dropped."""

    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _compute_axis_limits(
    neural_frame: pd.DataFrame,
    pinn_frame: pd.DataFrame,
    pinn_curve_frame: pd.DataFrame,
    trial_columns: list[str],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute stable x/y limits across all sequence panels."""

    x_values = [
        _numeric_series(neural_frame, "time"),
        _numeric_series(pinn_curve_frame, "time"),
    ]
    x_min = min(float(series.min()) for series in x_values if not series.empty)
    x_max = max(float(series.max()) for series in x_values if not series.empty)

    y_series = []
    for column in trial_columns:
        observed_column = f"{column}_observed"
        if observed_column in neural_frame.columns:
            y_series.append(_numeric_series(neural_frame, observed_column))
        elif observed_column in pinn_frame.columns:
            y_series.append(_numeric_series(pinn_frame, observed_column))

    for column in [
        "observed_mean",
        "classical_logistic_fitted_mean",
        "ode_fitted_mean",
        "sde_mean",
        "sde_lower",
        "sde_upper",
    ]:
        if column in neural_frame.columns:
            y_series.append(_numeric_series(neural_frame, column))

    for column in [
        "deterministic_pinn",
        "stochastic_pinn_drift",
        "sde_mean",
        "sde_lower",
        "sde_upper",
    ]:
        if column in pinn_curve_frame.columns:
            y_series.append(_numeric_series(pinn_curve_frame, column))

    y_min = min(float(series.min()) for series in y_series if not series.empty)
    y_max = max(float(series.max()) for series in y_series if not series.empty)

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    return (
        (x_min - 0.05 * x_range, x_max + 0.05 * x_range),
        (y_min - 0.05 * y_range, y_max + 0.05 * y_range),
    )


def _deduplicated_legend(axis) -> None:
    """Place one legend entry per label outside the plotting area."""

    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axis.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        fontsize=8.5,
    )


def save_all_model_comparison_sequence(
    neural_frame: pd.DataFrame,
    pinn_frame: pd.DataFrame,
    pinn_dense_frame: pd.DataFrame | None,
    trial_columns: list[str],
    interval_level: float,
    output_dir: Path,
) -> None:
    """Save step-by-step PNGs comparing all fitted models."""

    required_neural_columns = {
        "time",
        "observed_mean",
        "classical_logistic_fitted_mean",
        "ode_fitted_mean",
        "sde_mean",
        "sde_lower",
        "sde_upper",
    }
    missing_neural_columns = required_neural_columns.difference(neural_frame.columns)
    if missing_neural_columns:
        missing = ", ".join(sorted(missing_neural_columns))
        raise ValueError(f"Missing neural comparison columns: {missing}")

    required_pinn_columns = {
        "time",
        "deterministic_pinn",
        "stochastic_pinn_drift",
        "sde_mean",
        "sde_lower",
        "sde_upper",
    }
    pinn_curve_frame = _choose_pinn_frame(pinn_frame, pinn_dense_frame)
    missing_pinn_columns = required_pinn_columns.difference(pinn_curve_frame.columns)
    if missing_pinn_columns:
        missing = ", ".join(sorted(missing_pinn_columns))
        raise ValueError(f"Missing PINN comparison columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=(11.5, 5.9))
    fig.subplots_adjust(right=0.68, bottom=0.15)
    x_limits, y_limits = _compute_axis_limits(
        neural_frame,
        pinn_frame,
        pinn_curve_frame,
        trial_columns,
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_xlabel("Time (hrs)")
    axis.set_ylabel(r"Kefir wet biomass $(\mathrm{g/L})$")
    axis.grid(alpha=0.25)

    neural_time = neural_frame["time"]
    pinn_time = pinn_curve_frame["time"]

    def save_step(step_number: int, suffix: str, title: str) -> None:
        axis.set_title(title)
        _deduplicated_legend(axis)
        path = output_dir / f"{step_number:02d}_all_models_{suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved step {step_number}: {path}")

    for index, column in enumerate(trial_columns):
        observed_column = f"{column}_observed"
        source_frame = neural_frame if observed_column in neural_frame.columns else pinn_frame
        axis.plot(
            source_frame["time"],
            source_frame[observed_column],
            marker=TRIAL_MARKERS[index % len(TRIAL_MARKERS)],
            linestyle="",
            markersize=9.0,
            color=TRIAL_COLORS[index % len(TRIAL_COLORS)],
            markeredgecolor="black",
            alpha=0.45,
            label=column,
        )
    save_step(1, "data", "Water Kefir: Observed Trials")

    axis.plot(
        neural_time,
        neural_frame["observed_mean"],
        color=COLORS["observed_mean"],
        linewidth=2.0,
        linestyle=":",
        label="Observed mean",
    )
    save_step(2, "observed_mean", "Water Kefir: Observed Mean")

    axis.plot(
        neural_time,
        neural_frame["classical_logistic_fitted_mean"],
        color=COLORS["classical_logistic"],
        linewidth=2.5,
        linestyle=":",
        label="Classical logistic ODE",
    )
    save_step(3, "classical_logistic_ode", "Water Kefir: Classical Logistic ODE")

    axis.plot(
        neural_time,
        neural_frame["ode_fitted_mean"],
        color=COLORS["neural_ode"],
        linewidth=2.5,
        linestyle="--",
        label="Neural ODE",
    )
    save_step(4, "neural_ode", "Water Kefir: Neural ODE")

    axis.plot(
        neural_time,
        neural_frame["sde_mean"],
        color=COLORS["neural_sde"],
        linewidth=2.5,
        linestyle="-.",
        label="Neural SDE mean",
    )
    save_step(5, "neural_sde_mean", "Water Kefir: Neural SDE Mean")

    axis.fill_between(
        neural_time,
        neural_frame["sde_lower"],
        neural_frame["sde_upper"],
        color=COLORS["neural_sde_band"],
        alpha=0.15,
        label=f"Neural SDE {interval_level:.0%} interval",
    )
    axis.plot(
        neural_time,
        neural_frame["sde_lower"],
        color=COLORS["neural_sde_band"],
        linewidth=0.9,
        alpha=0.75,
    )
    axis.plot(
        neural_time,
        neural_frame["sde_upper"],
        color=COLORS["neural_sde_band"],
        linewidth=0.9,
        alpha=0.75,
    )
    save_step(6, "neural_sde_band", "Water Kefir: Classical and Neural Models")

    axis.plot(
        pinn_time,
        pinn_curve_frame["deterministic_pinn"],
        color=COLORS["deterministic_pinn"],
        linewidth=2.4,
        linestyle="-",
        label="Deterministic logistic PINN",
    )
    save_step(7, "deterministic_pinn", "Water Kefir: Add Deterministic PINN")

    axis.plot(
        pinn_time,
        pinn_curve_frame["stochastic_pinn_drift"],
        color=COLORS["stochastic_pinn_drift"],
        linewidth=2.4,
        linestyle="--",
        label="Stochastic logistic PINN drift",
    )
    save_step(8, "stochastic_pinn_drift", "Water Kefir: Add Stochastic PINN Drift")

    axis.plot(
        pinn_time,
        pinn_curve_frame["sde_mean"],
        color=COLORS["logistic_pinn_sde"],
        linewidth=2.4,
        linestyle="-.",
        label="Logistic PINN SDE mean",
    )
    save_step(9, "logistic_pinn_sde_mean", "Water Kefir: Add Logistic PINN SDE Mean")

    axis.fill_between(
        pinn_time,
        pinn_curve_frame["sde_lower"],
        pinn_curve_frame["sde_upper"],
        color=COLORS["logistic_pinn_sde_band"],
        alpha=0.14,
        label=f"Logistic PINN SDE {interval_level:.0%} interval",
    )
    axis.plot(
        pinn_time,
        pinn_curve_frame["sde_lower"],
        color=COLORS["logistic_pinn_sde_band"],
        linewidth=0.9,
        alpha=0.8,
    )
    axis.plot(
        pinn_time,
        pinn_curve_frame["sde_upper"],
        color=COLORS["logistic_pinn_sde_band"],
        linewidth=0.9,
        alpha=0.8,
    )
    save_step(10, "logistic_pinn_sde_band", "Water Kefir: All Fitted Models")

    final_path = output_dir / "water_kefir_all_models_comparison.png"
    fig.savefig(final_path, dpi=300, bbox_inches="tight")
    print(f"Saved final all-model comparison to {final_path}")
    plt.close(fig)


def main() -> None:
    """Run the all-model plotting sequence from saved output CSVs."""

    args = parse_args()
    if not args.neural_comparison_csv.exists():
        raise FileNotFoundError(
            f"Neural comparison CSV not found: {args.neural_comparison_csv}",
        )
    if not args.pinn_comparison_csv.exists():
        raise FileNotFoundError(
            f"PINN comparison CSV not found: {args.pinn_comparison_csv}",
        )

    neural_frame = pd.read_csv(args.neural_comparison_csv)
    pinn_frame = pd.read_csv(args.pinn_comparison_csv)
    pinn_dense_frame = (
        pd.read_csv(args.pinn_dense_csv) if args.pinn_dense_csv.exists() else None
    )

    trial_columns = infer_trial_columns(neural_frame)
    if not trial_columns:
        trial_columns = infer_trial_columns(pinn_frame)
    if not trial_columns:
        raise ValueError("Could not infer trial columns from the comparison CSVs.")

    save_all_model_comparison_sequence(
        neural_frame=neural_frame,
        pinn_frame=pinn_frame,
        pinn_dense_frame=pinn_dense_frame,
        trial_columns=trial_columns,
        interval_level=args.interval_level,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
