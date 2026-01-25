# 5_ablate.py
"""K-fold stress test."""

import csv
import datetime
import pathlib
import typing

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import fives_shared as fs

DiagramArray = fs.DiagramArray

TopoVector = typing.Annotated[np.ndarray, "9"]
TopoMatrix = typing.Annotated[np.ndarray, "N,9"]
GeoMatrix = typing.Annotated[np.ndarray, "N,7"]
LabelArray = typing.Annotated[np.ndarray, "N"]

CacheItem = typing.Tuple[
    typing.Tuple[DiagramArray, DiagramArray, DiagramArray], np.ndarray, int
]

PROTOCOL_MAP: typing.Dict[str, str] = {
    "gamma": "Gamma",
    "contrast": "Contrast",
    "drift": "Illumination Drift",
    "rotation": "Rotation",
    "blur": "Blur",
    "gau_noise": "Gaussian Noise",
    "poi_noise": "Poisson Noise",
    "spepper_noise": "Salt & Pepper",
    "bit_depth": "Bit Depth",
    "resolution": "Resolution",
}
FEATURE_SETS: typing.Dict[str, typing.Sequence[int]] = {
    "H0": (0, 1, 2),
    "H1": (3, 4, 5),
    "HS": (6, 7, 8),
    "H0H1": (0, 1, 2, 3, 4, 5),
    "H0HS": (0, 1, 2, 6, 7, 8),
    "H1HS": (3, 4, 5, 6, 7, 8),
    "H0H1HS": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "Geometric": (9, 10, 11, 12, 13, 14, 15),
}
CSV_COLUMNS: typing.Sequence[str] = tuple(
    ["Regime", "Parameter"]
    + [f"Acc_{name}" for name in FEATURE_SETS]
)


def _metric_key(name: str) -> str:
    """Return the perf log metric key.

    Parameters
    ----------
    name : str
        Feature set name.

    Returns
    -------
    str
        Perf log key for the feature set.

    """
    return "acc_{name}".format(name=name.lower())


def build_feature_matrices(
    items: typing.Sequence[CacheItem],
) -> typing.Tuple[TopoMatrix, GeoMatrix, LabelArray]:
    """Build topological and geometric matrices.

    Parameters
    ----------
    items : typing.Sequence[CacheItem]
        Cached diagrams, geometry vectors, and labels.

    Returns
    -------
    tuple of numpy.ndarray
        Topological matrix, geometric matrix, and labels.

    """
    topo_rows: typing.List[TopoVector] = []
    geo_rows: typing.List[np.ndarray] = []
    labels: typing.List[int] = []

    for diagrams, geom_vec, label in items:
        d0_sub, d1_sub, *rest = diagrams
        d0_sup = (
            rest[0]
            if rest
            else np.empty((0, 2), dtype=np.float32)
        )

        d0_array = np.asarray(d0_sub, dtype=np.float32).reshape(-1, 2)
        d1_array = np.asarray(d1_sub, dtype=np.float32).reshape(-1, 2)
        d0_sup_array = np.asarray(d0_sup, dtype=np.float32).reshape(-1, 2)

        h0_sub = fs.extract_topological_features(d0_array)
        h1_sub = fs.extract_topological_features(d1_array)
        h0_sup = fs.extract_topological_features(d0_sup_array)
        topo_vec = np.concatenate((h0_sub, h1_sub, h0_sup)).astype(
            np.float32
        )

        geo_vec = np.asarray(geom_vec, dtype=np.float32).reshape(-1)
        if geo_vec.shape[0] != 7:
            raise ValueError("Geometric vector length mismatch.")

        topo_rows.append(topo_vec)
        geo_rows.append(geo_vec)
        labels.append(int(label))

    topo_matrix = np.vstack(topo_rows).astype(np.float32)
    geo_matrix = np.vstack(geo_rows).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    return topo_matrix, geo_matrix, label_array


def load_cache_split(
    split: str,
    perturbation_type: str,
    level: typing.Union[int, float],
) -> typing.Tuple[TopoMatrix, GeoMatrix, LabelArray]:
    """Load cached features for a split and perturbation.

    Parameters
    ----------
    split : str
        Dataset split name.
    perturbation_type : str
        Perturbation protocol name.
    level : int or float
        Perturbation level.

    Returns
    -------
    tuple of numpy.ndarray
        Topological matrix, geometric matrix, and labels.

    """
    cache_path = fs.cache_path(split, perturbation_type, level)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")

    items = typing.cast(
        typing.List[CacheItem], fs.read_cache_stream(cache_path)
    )
    if not items:
        raise ValueError("Empty cache stream.")

    return build_feature_matrices(items)


def load_standard_total(
) -> typing.Tuple[TopoMatrix, GeoMatrix, LabelArray, int, int]:
    """Load standard train and test caches into aligned matrices.

    Returns
    -------
    tuple
        Topological matrix, geometric matrix, labels, train count, test count.

    """
    train_topo, train_geo, train_labels = load_cache_split(
        "train", "standard", 0
    )
    test_topo, test_geo, test_labels = load_cache_split("test", "standard", 0)

    topo_total = np.vstack((train_topo, test_topo)).astype(np.float32)
    geo_total = np.vstack((train_geo, test_geo)).astype(np.float32)
    label_total = np.concatenate((train_labels, test_labels)).astype(np.int64)

    return (
        topo_total,
        geo_total,
        label_total,
        train_topo.shape[0],
        test_topo.shape[0],
    )


def load_perturbed_total(
    perturbation_type: str,
    level: typing.Union[int, float],
    train_count: int,
    test_count: int,
) -> typing.Tuple[TopoMatrix, GeoMatrix, LabelArray]:
    """Load perturbed train and test caches into aligned matrices.

    Parameters
    ----------
    perturbation_type : str
        Perturbation protocol name.
    level : int or float
        Perturbation level.
    train_count : int
        Expected train item count.
    test_count : int
        Expected test item count.

    Returns
    -------
    tuple of numpy.ndarray
        Topological matrix, geometric matrix, and labels.

    """
    train_topo, train_geo, train_labels = load_cache_split(
        "train", perturbation_type, level
    )
    test_topo, test_geo, test_labels = load_cache_split(
        "test", perturbation_type, level
    )

    if train_topo.shape[0] != train_count or test_topo.shape[0] != test_count:
        raise ValueError("Perturbed cache size mismatch.")

    topo_total = np.vstack((train_topo, test_topo)).astype(np.float32)
    geo_total = np.vstack((train_geo, test_geo)).astype(np.float32)
    label_total = np.concatenate((train_labels, test_labels)).astype(np.int64)

    return topo_total, geo_total, label_total


def train_model(
    features: np.ndarray,
    labels: LabelArray,
) -> typing.Tuple[StandardScaler, LogisticRegression]:
    """Fit a standardised logistic regression model.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    labels : numpy.ndarray
        Label vector.

    Returns
    -------
    tuple
        Fitted scaler and logistic regression model.

    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features).astype(np.float32)

    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        random_state=fs.GLOBAL_SEED,
    )
    model.fit(scaled, labels)

    return scaler, model


def evaluate_model(
    features: np.ndarray,
    labels: LabelArray,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> float:
    """Return accuracy for a fitted model.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix to evaluate.
    labels : numpy.ndarray
        Label vector.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for standardization.
    model : sklearn.linear_model.LogisticRegression
        Fitted classifier.

    Returns
    -------
    float
        Classification accuracy.

    """
    scaled = scaler.transform(features).astype(np.float32)
    predictions = model.predict(scaled)
    return float(accuracy_score(labels, predictions))


def print_baseline_table(mean_scores: typing.Dict[str, float]) -> None:
    """Print the baseline accuracy table for standard data.

    Parameters
    ----------
    mean_scores : dict of str to float
        Mean accuracy per feature set.

    Returns
    -------
    None

    """
    headers = ["Baseline"] + [f"Acc {name}" for name in FEATURE_SETS]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    values = ["Standard"] + [
        fs.format_value(mean_scores[name]) for name in FEATURE_SETS
    ]
    print("| " + " | ".join(values) + " |")


def print_protocol_table(
    protocol_name: str,
    rows: typing.Sequence[typing.Tuple[str, typing.Dict[str, float]]],
) -> None:
    """Print the table for a perturbation protocol.

    Parameters
    ----------
    protocol_name : str
        Display name for the protocol.
    rows : typing.Sequence[tuple]
        Sequence of (level, scores) rows.

    Returns
    -------
    None

    """
    headers = ["Level"] + [f"Acc {name}" for name in FEATURE_SETS]
    print(f"\n### {protocol_name}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for level, scores in rows:
        values = [level] + [
            fs.format_value(scores[name]) for name in FEATURE_SETS
        ]
        print("| " + " | ".join(values) + " |")


def format_row(row: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """Format numeric row values to three decimals.

    Parameters
    ----------
    row : dict
        CSV row with numeric entries.

    Returns
    -------
    dict
        Row with formatted numeric values.

    """
    return {key: fs.format_value(value) for key, value in row.items()}


def run_full_kfold() -> None:
    """Run the k-fold evaluation and log results.

    Returns
    -------
    None

    """
    standard_topo, standard_geo, labels, train_count, test_count = (
        load_standard_total()
    )
    standard_all = np.hstack((standard_topo, standard_geo)).astype(np.float32)
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    perturbation_data: typing.Dict[
        typing.Tuple[str, typing.Union[int, float]],
        np.ndarray,
    ] = {}

    for perturbation_type, levels in fs.PROTOCOLS:
        if perturbation_type == "standard":
            continue
        for level in levels:
            topo_total, geo_total, pert_labels = load_perturbed_total(
                perturbation_type, level, train_count, test_count
            )
            if not np.array_equal(labels, pert_labels):
                raise ValueError("Label order mismatch across perturbations.")
            pert_all = np.hstack((topo_total, geo_total)).astype(np.float32)
            perturbation_data[(perturbation_type, level)] = pert_all

    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=fs.GLOBAL_SEED
    )

    baseline_accs: typing.Dict[str, typing.List[float]] = {
        name: [] for name in FEATURE_SETS
    }
    protocol_results: typing.Dict[
        typing.Tuple[str, typing.Union[int, float]],
        typing.Dict[str, typing.List[float]],
    ] = {
        key: {name: [] for name in FEATURE_SETS} for key in perturbation_data
    }

    for train_idx, val_idx in splitter.split(standard_all, labels):
        scalers: typing.Dict[str, StandardScaler] = {}
        models: typing.Dict[str, LogisticRegression] = {}

        for name, indices in FEATURE_SETS.items():
            train_features = standard_all[train_idx][:, indices]
            scaler, model = train_model(train_features, labels[train_idx])
            scalers[name] = scaler
            models[name] = model

            val_features = standard_all[val_idx][:, indices]
            acc = evaluate_model(val_features, labels[val_idx], scaler, model)
            baseline_accs[name].append(acc)

        for key, pert_all in perturbation_data.items():
            for name, indices in FEATURE_SETS.items():
                val_features = pert_all[val_idx][:, indices]
                acc = evaluate_model(
                    val_features, labels[val_idx], scalers[name], models[name]
                )
                protocol_results[key][name].append(acc)

    mean_baseline = {
        name: float(np.mean(values)) for name, values in baseline_accs.items()
    }
    print_baseline_table(mean_baseline)
    baseline_record = {
        "step": "5_ablate",
        "run_id": run_id,
        "protocol": "baseline",
        "folds": int(splitter.n_splits),
    }
    for name in FEATURE_SETS:
        baseline_record[_metric_key(name)] = mean_baseline[name]
    fs.append_perf_log(baseline_record)

    baseline_row = format_row(
        {
            "Regime": "baseline",
            "Parameter": "Standard",
            **{f"Acc_{name}": mean_baseline[name] for name in FEATURE_SETS},
        }
    )
    csv_rows: typing.List[typing.Dict[str, typing.Any]] = [baseline_row]
    per_protocol_rows: typing.Dict[
        str, typing.List[typing.Dict[str, typing.Any]]
    ] = {
        perturbation_type: [baseline_row.copy()]
        for perturbation_type, _ in fs.PROTOCOLS
        if perturbation_type != "standard"
    }

    for perturbation_type, levels in fs.PROTOCOLS:
        if perturbation_type == "standard":
            continue
        protocol_name = PROTOCOL_MAP.get(perturbation_type, perturbation_type)
        rows: typing.List[typing.Tuple[str, typing.Dict[str, float]]] = []

        for level in levels:
            key = (perturbation_type, level)
            results = protocol_results[key]
            mean_scores = {
                name: float(np.mean(results[name])) for name in FEATURE_SETS
            }

            rows.append((fs.format_value(level), mean_scores))
            record = {
                "step": "5_ablate",
                "run_id": run_id,
                "protocol": perturbation_type,
                "level": float(level),
                "folds": int(splitter.n_splits),
            }
            for name in FEATURE_SETS:
                record[_metric_key(name)] = mean_scores[name]
            fs.append_perf_log(record)
            csv_rows.append(
                format_row(
                    {
                        "Regime": perturbation_type,
                        "Parameter": float(level),
                        **{
                            f"Acc_{name}": mean_scores[name]
                            for name in FEATURE_SETS
                        },
                    }
                )
            )
            per_protocol_rows[perturbation_type].append(
                format_row(
                    {
                        "Regime": perturbation_type,
                        "Parameter": float(level),
                        **{
                            f"Acc_{name}": mean_scores[name]
                            for name in FEATURE_SETS
                        },
                    }
                )
            )

        print_protocol_table(protocol_name, rows)

    output_path = pathlib.Path("stress_test_results.csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved stress test summary to {output_path}")

    for perturbation_type, rows in per_protocol_rows.items():
        output_path = pathlib.Path(
            "stress_test_results_{ptype}.csv".format(ptype=perturbation_type)
        )
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved stress test summary to {output_path}")


if __name__ == "__main__":
    run_full_kfold()
