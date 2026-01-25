# 2_audit.py
"""Feature audit."""

import datetime
import pathlib
import typing

import numpy as np
from scipy import stats

import fives_shared as fs

DiagramArray = fs.DiagramArray

TopoVector = typing.Annotated[np.ndarray, "3"]
GeoVector = typing.Annotated[np.ndarray, "7"]
CombinedVector = typing.Annotated[np.ndarray, "16"]
FeatureMatrix = typing.Annotated[np.ndarray, "N,16"]
LabelArray = typing.Annotated[np.ndarray, "N"]

CacheItem = typing.Tuple[
    typing.Tuple[DiagramArray, DiagramArray, DiagramArray], np.ndarray, int
]

FEATURE_NAMES: typing.Sequence[str] = (
    "H0 Count",
    "H0 Max",
    "H0 Total",
    "H1 Count",
    "H1 Max",
    "H1 Total",
    "HS Count",
    "HS Max",
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
    cache_path: pathlib.Path,
) -> typing.Tuple[FeatureMatrix, LabelArray]:
    """Build a 16D feature matrix and labels.

    Parameters
    ----------
    cache_path : pathlib.Path
        Cache stream path.

    Returns
    -------
    tuple of numpy.ndarray
        Feature matrix and label array.

    """
    items = typing.cast(
        typing.List[CacheItem], fs.read_cache_stream(cache_path)
    )
    if not items:
        raise ValueError("Empty cache stream.")

    rows: typing.List[CombinedVector] = []
    labels: typing.List[int] = []

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
            raise ValueError("Geometric vector length mismatch.")

        combined_vec = np.concatenate((topo_vec, geo_vec)).astype(np.float32)
        rows.append(combined_vec)
        labels.append(int(label))

    feature_matrix = np.vstack(rows).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    return feature_matrix, label_array


def compute_feature_statistics(
    features: FeatureMatrix,
    labels: LabelArray,
    feature_names: typing.Sequence[str],
) -> typing.List[typing.Tuple[str, float, float, float, float]]:
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

    rows: typing.List[typing.Tuple[str, float, float, float, float]] = []

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
    rows: typing.Sequence[typing.Tuple[str, float, float, float, float]]
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
    cache_path = fs.cache_path("train", "standard", 0)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")

    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    features, labels = build_feature_matrix(cache_path)
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


if __name__ == "__main__":
    main()
