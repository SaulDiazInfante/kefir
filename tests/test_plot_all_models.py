import pandas as pd

from kefir_models import plot_all_models


def test_parse_args_help(capsys):
    """The all-model plotting CLI exposes help text."""

    try:
        plot_all_models.parse_args(["--help"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "step-by-step comparison across all kefir models" in captured.out


def test_infer_trial_columns():
    """Trial names are inferred from observed columns."""

    frame = pd.DataFrame(
        {
            "time": [0.0],
            "trial_1_observed": [1.0],
            "trial_2_observed": [2.0],
            "observed_mean": [1.5],
        },
    )

    assert plot_all_models.infer_trial_columns(frame) == ["trial_1", "trial_2"]
