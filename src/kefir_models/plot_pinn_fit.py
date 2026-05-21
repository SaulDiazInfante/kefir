"""Dedicated plotting script for the logistic PINN fits.

Produces two figures:

1. A step-by-step overlay plot (one PNG per step) showing:
   - Observed trial data
   - Deterministic PINN fitted curve
   - Stochastic PINN drift curve
   - Stochastic-PINN SDE mean + confidence band

2. A side-by-side training-loss figure for the deterministic and
   stochastic PINN models.

Usage (from project root)::

    kefir-plot-pinn-fit \\
        --comparison-csv  logistic_pinn_outputs/water_kefir_logistic_pinn_comparison.csv \\
        --dense-csv       logistic_pinn_outputs/water_kefir_logistic_pinn_dense.csv \\
        --det-loss-csv    logistic_pinn_outputs/deterministic_logistic_pinn_training_loss.csv \\
        --sto-loss-csv    logistic_pinn_outputs/stochastic_logistic_pinn_training_loss.csv \\
        --output-dir      logistic_pinn_outputs \\
        --interval-level  0.90
"""

import argparse
import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
_MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X"]


def _infer_trial_columns(frame: pd.DataFrame) -> list[str]:
    """Return trial base names from columns ending with '_observed'."""
    return [
        col.replace("_observed", "")
        for col in frame.columns
        if col.endswith("_observed") and col != "observed_mean"
    ]


# ---------------------------------------------------------------------------
# Step-by-step comparison plot
# ---------------------------------------------------------------------------

def save_pinn_comparison_plot(
    comparison_frame: pd.DataFrame,
    dense_frame: pd.DataFrame | None,
    trial_columns: list[str],
    interval_level: float,
    output_dir: Path,
) -> None:
    """Save observed data + PINN fits as step-by-step PNG files."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(right=0.65, bottom=0.15)

    obs_time = comparison_frame["time"]

    # Use the dense grid for smooth curves when available; fall back to
    # the observation time grid.
    if dense_frame is not None and not dense_frame.empty:
        smooth_time = dense_frame["time"]
        det_curve = dense_frame.get("deterministic_pinn", comparison_frame.get("deterministic_pinn"))
        sto_drift = dense_frame.get("stochastic_pinn_drift", comparison_frame.get("stochastic_pinn_drift"))
        sde_mean = dense_frame.get("sde_mean", comparison_frame.get("sde_mean"))
        sde_lower = dense_frame.get("sde_lower", comparison_frame.get("sde_lower"))
        sde_upper = dense_frame.get("sde_upper", comparison_frame.get("sde_upper"))
    else:
        smooth_time = obs_time
        det_curve = comparison_frame.get("deterministic_pinn")
        sto_drift = comparison_frame.get("stochastic_pinn_drift")
        sde_mean = comparison_frame.get("sde_mean")
        sde_lower = comparison_frame.get("sde_lower")
        sde_upper = comparison_frame.get("sde_upper")

    # Pre-compute axis limits for a stable layout across all steps
    all_y_cols = ["observed_mean", "deterministic_pinn", "stochastic_pinn_drift",
                  "sde_mean", "sde_lower", "sde_upper"]
    for col in trial_columns:
        all_y_cols.append(f"{col}_observed")
    valid_cols = [c for c in all_y_cols if c in comparison_frame.columns]
    y_min = comparison_frame[valid_cols].min().min()
    y_max = comparison_frame[valid_cols].max().max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    axis.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    x_min, x_max = obs_time.min(), obs_time.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    axis.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    axis.set_xlabel("Time (hrs)")
    axis.set_ylabel(r"Kefir wet biomass $(\mathrm{g/L})$")
    axis.grid(alpha=0.25)

    def _save_step(step_num: int, suffix: str, title: str) -> None:
        axis.set_title(title)
        handles, labels = axis.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axis.legend(
            by_label.values(), by_label.keys(),
            bbox_to_anchor=(1.05, 1), loc="upper left",
            fontsize=9,
        )
        step_path = output_dir / f"{step_num:02d}_pinn_fit_{suffix}.png"
        fig.savefig(step_path, dpi=300, bbox_inches="tight")
        print(f"Saved step {step_num}: {step_path}")

    # ------------------------------------------------------------------
    # Step 1 – observed data
    # ------------------------------------------------------------------
    for idx, col in enumerate(trial_columns):
        col_obs = f"{col}_observed"
        if col_obs not in comparison_frame.columns:
            continue
        axis.plot(
            obs_time,
            comparison_frame[col_obs],
            marker=_MARKERS[idx % len(_MARKERS)],
            linestyle="",
            markersize=10,
            color=_COLORS[idx % len(_COLORS)],
            markeredgecolor="black",
            alpha=0.65,
            label=col.replace("_", " ").title(),
        )
    _save_step(1, "data", "Water Kefir: Observed Trials")

    # ------------------------------------------------------------------
    # Step 2 – deterministic PINN fit
    # ------------------------------------------------------------------
    if det_curve is not None:
        axis.plot(
            smooth_time,
            det_curve,
            color="#2364aa",
            linewidth=2.5,
            linestyle="-",
            label="Deterministic PINN",
        )
    _save_step(2, "det_pinn", "Water Kefir: Deterministic PINN Fit")

    # ------------------------------------------------------------------
    # Step 3 – stochastic PINN drift
    # ------------------------------------------------------------------
    if sto_drift is not None:
        axis.plot(
            smooth_time,
            sto_drift,
            color="#e76f51",
            linewidth=2.5,
            linestyle="--",
            label="Stochastic PINN drift",
        )
    _save_step(3, "sto_pinn_drift", "Water Kefir: Stochastic PINN Drift")

    # ------------------------------------------------------------------
    # Step 4 – SDE mean + confidence band
    # ------------------------------------------------------------------
    if sde_mean is not None:
        axis.plot(
            smooth_time,
            sde_mean,
            color="#d95f02",
            linewidth=2.5,
            linestyle="-.",
            label="SDE mean",
        )
    if sde_lower is not None and sde_upper is not None:
        axis.fill_between(
            smooth_time,
            sde_lower,
            sde_upper,
            color="#d95f02",
            alpha=0.18,
            label=f"SDE {interval_level:.0%} band",
        )
        axis.plot(smooth_time, sde_lower, color="#d95f02", linewidth=0.8, alpha=0.5)
        axis.plot(smooth_time, sde_upper, color="#d95f02", linewidth=0.8, alpha=0.5)
    _save_step(4, "sde_band", "Water Kefir: Logistic PINN Fits")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training-loss figure
# ---------------------------------------------------------------------------

def save_loss_plot(
    det_loss: pd.DataFrame | None,
    sto_loss: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Save deterministic and stochastic PINN training-loss histories."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    titles = ["Deterministic PINN", "Stochastic PINN"]
    frames = [det_loss, sto_loss]

    # Columns to plot per model (skip epoch and the learned parameters)
    _skip = {"epoch", "growth_rate", "carrying_capacity", "diffusion"}

    for ax, title, frame in zip(axes, titles, frames):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)

        if frame is None or frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        x_col = "epoch" if "epoch" in frame.columns else None
        x = (pd.to_numeric(frame[x_col], errors="coerce")
             if x_col else pd.Series(range(1, len(frame) + 1)))

        loss_cols = [
            c for c in frame.columns
            if c not in _skip and pd.api.types.is_numeric_dtype(frame[c])
        ]

        all_vals = []
        for col in loss_cols:
            y = pd.to_numeric(frame[col], errors="coerce")
            valid = x.notna() & y.notna() & (y > 0)
            if not valid.any():
                continue
            all_vals.append(y[valid])
            ax.plot(x[valid], y[valid], linewidth=1.6,
                    label=col.replace("_", " ").title())

        if all_vals:
            ax.set_yscale("log")
            ax.legend(fontsize=8)

    fig.suptitle("Logistic PINN: Training Loss Histories", fontsize=13)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Loss plot saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot logistic PINN fitting results (step-by-step comparison + loss histories).",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/water_kefir_logistic_pinn_comparison.csv"),
        help="Path to the per-observation-time comparison CSV.",
    )
    parser.add_argument(
        "--dense-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/water_kefir_logistic_pinn_dense.csv"),
        help="Path to the dense-grid prediction CSV (for smooth curves).",
    )
    parser.add_argument(
        "--det-loss-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/deterministic_logistic_pinn_training_loss.csv"),
        help="Deterministic PINN training-loss CSV.",
    )
    parser.add_argument(
        "--sto-loss-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/stochastic_logistic_pinn_training_loss.csv"),
        help="Stochastic PINN training-loss CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logistic_pinn_outputs"),
        help="Directory where output PNGs are written.",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
        help="Confidence-band level shown in the legend label (default: 0.90).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point called by kefir-plot-pinn-fit."""
    args = parse_args()

    if not args.comparison_csv.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {args.comparison_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_frame = pd.read_csv(args.comparison_csv)
    trial_columns = _infer_trial_columns(comparison_frame)
    if not trial_columns:
        raise ValueError("No trial columns found in the comparison CSV.")

    dense_frame = None
    if args.dense_csv.exists():
        dense_frame = pd.read_csv(args.dense_csv)

    det_loss = pd.read_csv(args.det_loss_csv) if args.det_loss_csv.exists() else None
    sto_loss = pd.read_csv(args.sto_loss_csv) if args.sto_loss_csv.exists() else None

    # --- comparison steps ---
    save_pinn_comparison_plot(
        comparison_frame,
        dense_frame,
        trial_columns,
        args.interval_level,
        args.output_dir,
    )

    # --- loss histories ---
    loss_path = args.output_dir / "pinn_training_losses.png"
    save_loss_plot(det_loss, sto_loss, loss_path)


if __name__ == "__main__":
    main()
