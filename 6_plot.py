# 6_plot.py
"""Plot stress-test curves."""

from __future__ import annotations

import json
import pathlib
import typing

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import fives_shared as fs

# Plot defaults
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 12

PerfRecord = typing.Dict[str, typing.Any]
SeriesMap = typing.Dict[str, typing.List[float]]

PROTOCOL_PLOTS: typing.Sequence[typing.Tuple[str, str, str]] = (
    (
        "drift",
        "Photometric Stability: Illumination Drift",
        "Illumination Offset",
    ),
    (
        "rotation",
        "Digital Stability: Rotation",
        "Rotation Angle (Degrees)",
    ),
    ("gamma", "Photometric Stability: Gamma", "Gamma"),
    ("contrast", "Photometric Stability: Contrast", "Contrast Factor"),
    ("blur", "Optical Stability: Blur", "Gaussian Sigma"),
    ("gau_noise", "Sensor Stability: Gaussian Noise", "Gaussian Sigma"),
    ("poi_noise", "Sensor Stability: Poisson Noise", "Severity"),
    ("spepper_noise", "Sensor Stability: Salt-and-Pepper", "Severity"),
    ("bit_depth", "Sensor Stability: Bit Depth", "Bit Depth (bits)"),
    ("resolution", "Sensor Stability: Resolution", "Resolution (px)"),
)
BASELINE_LEVELS: typing.Dict[str, float] = {
    "drift": 0.0,
    "rotation": 0.0,
    "gamma": 1.0,
    "contrast": 1.0,
    "blur": 0.0,
    "gau_noise": 0.0,
    "poi_noise": 0.0,
    "spepper_noise": 0.0,
    "bit_depth": 8.0,
    "resolution": 2048.0,
}
LEGEND_CONFIG: typing.Dict[str, typing.Dict[str, typing.Any]] = {
    "contrast": {"loc": "lower center", "ncol": 2},
    "gamma": {"loc": "lower center", "ncol": 2},
    "drift": {"loc": "lower center", "ncol": 2},
    "bit_depth": {"loc": "upper left", "ncol": 2},
    "resolution": {"loc": "lower center", "ncol": 2},
    "rotation": {"loc": "lower center", "ncol": 2},
    "blur": {"loc": "best", "ncol": 2},
    "gau_noise": {"loc": "best", "ncol": 2},
    "poi_noise": {"loc": "best", "ncol": 2},
    "spepper_noise": {"loc": "best", "ncol": 2},
}
SERIES_ORDER: typing.Sequence[str] = (
    "acc_h0",
    "acc_h1",
    "acc_hs",
    "acc_h0h1",
    "acc_h0hs",
    "acc_h1hs",
    "acc_h0h1hs",
    "acc_geometric",
)
SERIES_STYLES: typing.Dict[str, typing.Dict[str, typing.Any]] = {
    "acc_h0": {"label": "H0", "color": "#e06666"},
    "acc_h1": {"label": "H1", "color": "#6fa8dc"},
    "acc_hs": {"label": "HS", "color": "#f6b26b"},
    "acc_h0h1": {"label": "H0H1", "color": "#93c47d"},
    "acc_h0hs": {"label": "H0HS", "color": "#8e7cc3"},
    "acc_h1hs": {"label": "H1HS", "color": "#76a5af"},
    "acc_h0h1hs": {"label": "H0H1HS", "color": "#38761d"},
    "acc_geometric": {"label": "Geometric", "color": "#7f7f7f"},
}
LINESTYLE_MAP: typing.Dict[str, typing.Any] = {
    "acc_h0": "-",
    "acc_h1": "-",
    "acc_hs": "-",
    "acc_h0h1": (0, (3, 1, 1, 1)),
    "acc_h0hs": (0, (3, 1, 1, 1)),
    "acc_h1hs": (0, (3, 1, 1, 1)),
    "acc_h0h1hs": "-.",
    "acc_geometric": ":",
}


def load_perf_log(path: pathlib.Path) -> typing.List[PerfRecord]:
    """Load performance records from a JSONL log.

    Parameters
    ----------
    path : pathlib.Path
        JSONL log path.

    Returns
    -------
    list of dict
        Performance records.

    """
    if not path.exists():
        raise FileNotFoundError(f"Missing perf log: {path}")

    records: typing.List[PerfRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_latest_run(records: typing.Sequence[PerfRecord]) -> str:
    """Return the latest run_id.

    Parameters
    ----------
    records : typing.Sequence[dict]
        Performance records.

    Returns
    -------
    str
        Latest run identifier.

    """
    run_ids = [
        record.get("run_id") for record in records if record.get("run_id")
    ]
    if not run_ids:
        raise ValueError("Missing run_id values in perf log.")
    return max(run_ids)


def build_series(
    records: typing.Sequence[PerfRecord],
    protocol: str,
    baseline: typing.Optional[PerfRecord],
    baseline_level: typing.Optional[float],
) -> typing.Tuple[typing.List[float], SeriesMap, typing.List[str]]:
    """Build the accuracy series for one protocol.

    Parameters
    ----------
    records : typing.Sequence[dict]
        Performance records for a run.
    protocol : str
        Protocol name.
    baseline : dict or None
        Baseline record to inject if missing.
    baseline_level : float or None
        Baseline level to inject if missing.

    Returns
    -------
    tuple
        Levels, series map, and available keys.

    """
    proto_records = [
        record
        for record in records
        if record.get("protocol") == protocol and "level" in record
    ]
    if not proto_records:
        raise ValueError(f"Missing protocol data for {protocol}.")

    available = [key for key in SERIES_ORDER if key in proto_records[0]]
    if not available:
        raise ValueError("Missing accuracy keys in perf log records.")

    series = []
    for record in proto_records:
        if "level" not in record:
            continue
        point = {"level": float(record["level"])}
        for key in available:
            if key not in record:
                raise ValueError(
                    "Missing {key} in perf_log; rerun 5_ablate.".format(
                        key=key
                    )
                )
            point[key] = float(record[key]) * 100.0
        series.append(point)
    levels_present = {row["level"] for row in series}
    if (
        baseline is not None
        and baseline_level is not None
        and baseline_level not in levels_present
    ):
        point = {"level": float(baseline_level)}
        for key in available:
            if key not in baseline:
                raise ValueError(
                    "Missing {key} in perf_log baseline; rerun 5_ablate.".format(
                        key=key
                    )
                )
            point[key] = float(baseline[key]) * 100.0
        series.append(point)

    series.sort(key=lambda row: row["level"])
    levels = [row["level"] for row in series]
    series_map = {key: [row[key] for row in series] for key in available}

    return levels, series_map, available


def set_accuracy_limits(
    ax: plt.Axes, series: typing.Sequence[typing.List[float]]
) -> None:
    """Set y-axis limits from the data range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to update.
    series : typing.Sequence[typing.List[float]]
        Series values for limits.

    Returns
    -------
    None

    """
    values: typing.List[float] = []
    for values_list in series:
        values.extend(values_list)
    lower = max(0.0, min(values) - 5.0)
    upper = min(100.0, max(values) + 5.0)
    ax.set_ylim(lower, upper)


def build_bit_depth_ticks(
    levels: typing.Sequence[float],
) -> typing.Tuple[typing.List[float], typing.List[str]]:
    """Return tick positions and integer labels for bit depth.

    Parameters
    ----------
    levels : typing.Sequence[float]
        Bit depth levels.

    Returns
    -------
    tuple
        Tick positions and labels.

    """
    ticks: typing.List[float] = []
    labels: typing.List[str] = []
    for level in levels:
        if level <= 0:
            continue
        ticks.append(level)
        labels.append(str(int(round(level))))
    return ticks, labels


def plot_stress_tests() -> None:
    """Plot stress-test curves into separate figures.

    Returns
    -------
    None

    """
    records = load_perf_log(fs.PERF_LOG_PATH)
    step_records = [
        record
        for record in records
        if record.get("step") == "5_ablate"
    ]
    if not step_records:
        raise ValueError("Missing 5_ablate records in perf log.")

    run_id = select_latest_run(step_records)
    run_records = [
        record for record in step_records if record.get("run_id") == run_id
    ]
    baseline = next(
        (
            record
            for record in run_records
            if record.get("protocol") == "baseline"
        ),
        None,
    )

    output_dir = pathlib.Path("images")
    output_dir.mkdir(parents=True, exist_ok=True)

    for protocol, title, xlabel in PROTOCOL_PLOTS:
        try:
            levels, series_map, keys = build_series(
                run_records,
                protocol,
                baseline,
                BASELINE_LEVELS.get(protocol),
            )
        except ValueError:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.5))
        for key in keys:
            style = SERIES_STYLES.get(key, {})
            label = style.get("label", key)
            color = style.get("color")
            plot_kwargs = {
                "linewidth": 1,
                "linestyle": LINESTYLE_MAP.get(key, "-"),
                "label": label,
            }
            if key == "acc_geometric":
                plot_kwargs["linewidth"] = 2
            if color:
                plot_kwargs["color"] = color
            ax.plot(levels, series_map[key], **plot_kwargs)
        baseline_x = BASELINE_LEVELS.get(protocol)
        if baseline_x is not None:
            baseline_color = SERIES_STYLES.get(
                "acc_geometric", {}
            ).get("color", "#7f7f7f")
            ax.axvline(
                baseline_x,
                color=baseline_color,
                linestyle="--",
                linewidth=2,
                label="baseline",
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Classification Accuracy (%)")
        set_accuracy_limits(ax, series_map.values())
        ax.yaxis.set_major_locator(MultipleLocator(5))
        if protocol == "bit_depth":
            ticks, labels = build_bit_depth_ticks(levels)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks(levels)
        if protocol == "resolution":
            base_size = float(plt.rcParams.get("font.size", 12))
            ax.tick_params(axis="x", labelsize=base_size * 0.5)
        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle=":",
            alpha=0.6,
        )
        legend_kwargs = LEGEND_CONFIG.get(
            protocol, {"loc": "best", "ncol": 2}
        )
        ax.legend(
            fontsize=8.4,
            **legend_kwargs,
        )

        output_path = output_dir / f"stress_test_{protocol}.png"
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_stress_tests()
