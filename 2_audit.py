# 2_audit.py
"""Feature audit."""

import datetime
import pathlib
import typing

import numpy as np
from scipy import stats

import fives_shared as fs

DiagramArray = fs.DiagramArray

CombinedVector = typing.Annotated[np.ndarray, "13"]
FeatureMatrix = typing.Annotated[np.ndarray, "N,13"]
LabelArray = typing.Annotated[np.ndarray, "N"]

CacheItem = tuple[
    tuple[DiagramArray, DiagramArray, DiagramArray], np.ndarray, int
]

FEATURE_NAMES: typing.Sequence[str] = (
    "H0 Top5",
    "H0 Total",
    "H1 Top5",
    "H1 Total",
    "HS Top5",
    "HS Total",
    "Hu 1",
    "Hu 2",
    "Hu 3",
    "Hu 4",
    "Hu 5",
    "Hu 6",
    "Hu 7",
)


def build_feature_matrix(
    items: list[CacheItem],
) -> tuple[FeatureMatrix, LabelArray]:
    """Build a 13D feature matrix and labels.

    Parameters
    ----------
    items : list of CacheItem
        Cached diagrams, Hu moments, and labels.

    Returns
    -------
    tuple of numpy.ndarray
        Feature matrix and label array.

    """
    if not items:
        raise ValueError("Empty cache items.")

    rows: list[CombinedVector] = []
    labels: list[int] = []

    for diagrams, geom_vec, label in items:
        d0_sub, d1_sub, *rest = diagrams
        d0_sup = (
            rest[0] if rest else np.empty((0, 2), dtype=np.float32)
        )
        d0_array = np.asarray(d0_sub, dtype=np.float32).reshape(-1, 2)
        d1_array = np.asarray(d1_sub, dtype=np.float32).reshape(-1, 2)
        d0_sup_array = np.asarray(d0_sup, dtype=np.float32).reshape(-1, 2)

        h0_sub_vec = fs.extract_topological_features(d0_array)
        h1_sub_vec = fs.extract_topological_features(d1_array)
        h0_sup_vec = fs.extract_topological_features(d0_sup_array)
        topo_vec = np.concatenate(
            (h0_sub_vec, h1_sub_vec, h0_sup_vec)
        ).astype(np.float32)

        geo_vec = np.asarray(geom_vec, dtype=np.float32).reshape(-1)
        if geo_vec.shape[0] != 7:
            raise ValueError("Hu vector length mismatch.")

        combined_vec = np.concatenate((topo_vec, geo_vec)).astype(np.float32)
        rows.append(combined_vec)
        labels.append(int(label))

    feature_matrix = np.vstack(rows).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    return feature_matrix, label_array


def _lifetimes_from_diagram(diagram: DiagramArray) -> np.ndarray:
    """Return finite, non-negative lifetimes from a diagram."""
    if diagram.size == 0:
        return np.empty((0,), dtype=np.float32)
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    if finite.size == 0:
        return np.empty((0,), dtype=np.float32)
    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = np.maximum(lifetimes, 0.0)
    return lifetimes.astype(np.float32)


def _init_lifetime_stats() -> dict[str, float]:
    return {"count": 0.0, "sum": 0.0, "min": np.inf, "max": -np.inf}


def _update_lifetime_stats(stats: dict[str, float], lifetimes: np.ndarray) -> None:
    if lifetimes.size == 0:
        return
    stats["count"] += float(lifetimes.size)
    stats["sum"] += float(lifetimes.sum())
    stats["min"] = min(stats["min"], float(lifetimes.min()))
    stats["max"] = max(stats["max"], float(lifetimes.max()))


def _histogram_summary(
    hist: np.ndarray,
    edges: np.ndarray,
) -> tuple[float, float]:
    if hist.sum() == 0:
        return 0.0, 0.0
    cumulative = np.cumsum(hist)
    target = 0.5 * float(cumulative[-1])
    median_idx = int(np.searchsorted(cumulative, target, side="left"))
    median = float(0.5 * (edges[median_idx] + edges[median_idx + 1]))
    mode_idx = int(np.argmax(hist))
    mode = float(0.5 * (edges[mode_idx] + edges[mode_idx + 1]))
    return median, mode


def compute_lifetime_summary(
    items: list[CacheItem],
    bins: int = 256,
) -> dict[str, dict[str, float]]:
    """Compute mean, median, and mode of lifetimes per homology group.

    Median and mode are estimated from a binned histogram.

    Parameters
    ----------
    items : list of CacheItem
        Cached diagrams, Hu moments, and labels.
    bins : int, optional
        Histogram bins for median/mode estimation.

    Returns
    -------
    dict
        Mapping of homology name to mean/median/mode/count.
    """
    stats = {
        "H0": _init_lifetime_stats(),
        "H1": _init_lifetime_stats(),
        "HS": _init_lifetime_stats(),
    }

    for diagrams, _, _ in items:
        d0_sub, d1_sub, *rest = diagrams
        d0_sup = rest[0] if rest else np.empty((0, 2), dtype=np.float32)
        _update_lifetime_stats(stats["H0"], _lifetimes_from_diagram(d0_sub))
        _update_lifetime_stats(stats["H1"], _lifetimes_from_diagram(d1_sub))
        _update_lifetime_stats(stats["HS"], _lifetimes_from_diagram(d0_sup))

    summaries: dict[str, dict[str, float]] = {}
    histograms: dict[str, np.ndarray] = {}
    edges_map: dict[str, np.ndarray] = {}

    for key, stat in stats.items():
        count = int(stat["count"])
        if count == 0:
            summaries[key] = {
                "mean": 0.0,
                "median": 0.0,
                "mode": 0.0,
                "count": 0.0,
            }
            continue
        min_val = stat["min"]
        max_val = stat["max"]
        if min_val == max_val:
            summaries[key] = {
                "mean": float(stat["sum"] / stat["count"]),
                "median": float(min_val),
                "mode": float(min_val),
                "count": float(count),
            }
            continue
        edges = np.linspace(min_val, max_val, bins + 1)
        edges_map[key] = edges
        histograms[key] = np.zeros(bins, dtype=np.int64)

    for diagrams, _, _ in items:
        d0_sub, d1_sub, *rest = diagrams
        d0_sup = rest[0] if rest else np.empty((0, 2), dtype=np.float32)
        for key, diagram in (("H0", d0_sub), ("H1", d1_sub), ("HS", d0_sup)):
            if key not in histograms:
                continue
            lifetimes = _lifetimes_from_diagram(diagram)
            if lifetimes.size == 0:
                continue
            histograms[key] += np.histogram(
                lifetimes, bins=edges_map[key]
            )[0]

    for key, stat in stats.items():
        if key in summaries:
            continue
        mean = float(stat["sum"] / stat["count"])
        median, mode = _histogram_summary(histograms[key], edges_map[key])
        summaries[key] = {
            "mean": mean,
            "median": median,
            "mode": mode,
            "count": float(stat["count"]),
        }

    return summaries


def compute_feature_statistics(
    features: FeatureMatrix,
    labels: LabelArray,
    feature_names: typing.Sequence[str],
) -> list[tuple[str, float, float, float, float]]:
    """Compute per-feature group statistics.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    labels : numpy.ndarray
        Binary labels.
    feature_names : sequence of str
        Feature labels in column order.

    Returns
    -------
    list of tuple
        (name, healthy mean, diabetic mean, p-value, Cohen's d).

    """
    healthy_mask = labels == 0
    diabetic_mask = labels == 1

    if not healthy_mask.any() or not diabetic_mask.any():
        raise ValueError("Missing cohort labels in cache.")

    healthy = features[healthy_mask]
    diabetic = features[diabetic_mask]

    rows: list[tuple[str, float, float, float, float]] = []

    for idx, name in enumerate(feature_names):
        healthy_vals = healthy[:, idx]
        diabetic_vals = diabetic[:, idx]

        mean_healthy = float(np.mean(healthy_vals))
        mean_diabetic = float(np.mean(diabetic_vals))

        if healthy_vals.size < 2 or diabetic_vals.size < 2:
            p_value = 1.0
            cohens_d = 0.0
        else:
            t_result = stats.ttest_ind(
                healthy_vals, diabetic_vals, equal_var=False
            )
            p_value = (
                float(t_result.pvalue)
                if np.isfinite(t_result.pvalue)
                else 1.0
            )
            var_healthy = float(np.var(healthy_vals, ddof=1))
            var_diabetic = float(np.var(diabetic_vals, ddof=1))
            denom = np.sqrt(0.5 * (var_healthy + var_diabetic))
            cohens_d = (
                float((mean_diabetic - mean_healthy) / denom)
                if denom > 0 and np.isfinite(denom)
                else 0.0
            )

        rows.append((name, mean_healthy, mean_diabetic, p_value, cohens_d))

    return rows


def print_markdown_table(
    rows: typing.Sequence[tuple[str, float, float, float, float]]
) -> None:
    """Print a markdown table of feature statistics.

    Parameters
    ----------
    rows : sequence of tuple
        (name, healthy mean, diabetic mean, p-value, Cohen's d).

    Returns
    -------
    None

    """
    print("| Feature Name | Healthy Mean | Diabetic Mean | p-value | Cohen's d |")
    print("| --- | --- | --- | --- | --- |")
    for name, mean_healthy, mean_diabetic, p_value, cohens_d in rows:
        print(
            "| {name} | {mh} | {md} | {pv} | {cd} |".format(
                name=name,
                mh=fs.format_value(mean_healthy),
                md=fs.format_value(mean_diabetic),
                pv=fs.format_value(p_value),
                cd=fs.format_value(cohens_d),
            )
        )


def main() -> None:
    """Run the feature audit and log results.

    Returns
    -------
    None

    """
    fs.seed_everything()
    cache_path = fs.cache_path("train", "standard", 0)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")

    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")

    items = typing.cast(list[CacheItem], fs.read_cache_stream(cache_path))
    if not items:
        raise ValueError("Empty cache stream.")

    features, labels = build_feature_matrix(items)
    lifetime_summary = compute_lifetime_summary(items)
    rows = compute_feature_statistics(features, labels, FEATURE_NAMES)
    rows_sorted = sorted(rows, key=lambda row: abs(row[4]), reverse=True)
    print_markdown_table(rows_sorted)

    for name, mean_healthy, mean_diabetic, p_value, cohens_d in rows_sorted:
        fs.append_perf_log(
            {
                "step": "2_audit",
                "run_id": run_id,
                "feature": name,
                "mean_healthy": float(mean_healthy),
                "mean_diabetic": float(mean_diabetic),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "split": "train",
                "perturbation": "standard",
                "level": 0,
            }
        )

    print("\n| Homology | Mean | Median | Mode | Count |")
    print("| --- | --- | --- | --- | --- |")
    for key in ("H0", "H1", "HS"):
        summary = lifetime_summary[key]
        print(
            "| {name} | {mean} | {median} | {mode} | {count} |".format(
                name=key,
                mean=fs.format_value(summary["mean"]),
                median=fs.format_value(summary["median"]),
                mode=fs.format_value(summary["mode"]),
                count=fs.format_value(summary["count"]),
            )
        )
        fs.append_perf_log(
            {
                "step": "2_audit_lifetimes",
                "run_id": run_id,
                "homology": key,
                "mean": float(summary["mean"]),
                "median": float(summary["median"]),
                "mode": float(summary["mode"]),
                "count": float(summary["count"]),
                "split": "train",
                "perturbation": "standard",
                "level": 0,
            }
        )


if __name__ == "__main__":
    main()
