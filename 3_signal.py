# 3_signal.py
"""Feature ablation validation."""

import datetime
import pathlib
import typing

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import fives_shared as fs

DiagramArray = fs.DiagramArray

TopoVector = typing.Annotated[np.ndarray, "2"]
CombinedVector = typing.Annotated[np.ndarray, "6"]
TopoMatrix = typing.Annotated[np.ndarray, "N,2"]
CombinedMatrix = typing.Annotated[np.ndarray, "N,6"]
GeoMatrix = typing.Annotated[np.ndarray, "N,7"]
LabelArray = typing.Annotated[np.ndarray, "N"]

CacheItem = tuple[
    tuple[DiagramArray, DiagramArray, DiagramArray], np.ndarray, int
]


def build_feature_sets(
    cache_path: pathlib.Path,
) -> tuple[
    TopoMatrix, TopoMatrix, TopoMatrix, CombinedMatrix, GeoMatrix, LabelArray
]:
    """Build H0, H1, HS, H0H1HS, and Hu moment matrices.

    Parameters
    ----------
    cache_path : pathlib.Path
        Cache stream path.

    Returns
    -------
    tuple of numpy.ndarray
        H0, H1, HS, H0H1HS, Hu moment matrices, and labels.

    """
    items = typing.cast(list[CacheItem], fs.read_cache_stream(cache_path))
    if not items:
        raise ValueError("Empty cache stream.")

    h0_rows: list[TopoVector] = []
    h1_rows: list[TopoVector] = []
    hs_rows: list[TopoVector] = []
    combined_rows: list[CombinedVector] = []
    geo_rows: list[np.ndarray] = []
    labels: list[int] = []

    for diagrams, geom_vec, label in items:
        d0_sub, d1_sub, *rest = diagrams
        d0_sup = rest[0] if rest else np.empty((0, 2), dtype=np.float32)
        d0_array = np.asarray(d0_sub, dtype=np.float32).reshape(-1, 2)
        d1_array = np.asarray(d1_sub, dtype=np.float32).reshape(-1, 2)
        d0_sup_array = np.asarray(d0_sup, dtype=np.float32).reshape(-1, 2)

        h0_vec = fs.extract_topological_features(d0_array)
        h1_vec = fs.extract_topological_features(d1_array)
        hs_vec = fs.extract_topological_features(d0_sup_array)
        combined_vec = np.concatenate((h0_vec, h1_vec, hs_vec)).astype(
            np.float32
        )
        geo_vec = np.asarray(geom_vec, dtype=np.float32).reshape(-1)

        h0_rows.append(h0_vec)
        h1_rows.append(h1_vec)
        hs_rows.append(hs_vec)
        combined_rows.append(combined_vec)
        geo_rows.append(geo_vec)
        labels.append(label)

    h0_matrix = np.vstack(h0_rows).astype(np.float32)
    h1_matrix = np.vstack(h1_rows).astype(np.float32)
    hs_matrix = np.vstack(hs_rows).astype(np.float32)
    combined_matrix = np.vstack(combined_rows).astype(np.float32)
    geo_matrix = np.vstack(geo_rows).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    return (
        h0_matrix,
        h1_matrix,
        hs_matrix,
        combined_matrix,
        geo_matrix,
        label_array,
    )


def evaluate_condition(
    features: np.ndarray,
    labels: LabelArray,
) -> tuple[float, float]:
    """Return mean accuracy and AUC from stratified CV.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix.
    labels : numpy.ndarray
        Binary labels.

    Returns
    -------
    tuple of float
        Mean accuracy and mean AUC.

    """
    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=fs.GLOBAL_SEED
    )
    accuracies: list[float] = []
    aucs: list[float] = []

    for train_idx, val_idx in splitter.split(features, labels):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_idx]).astype(
            np.float32
        )
        val_features = scaler.transform(features[val_idx]).astype(np.float32)

        model = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            penalty="l2",
            random_state=fs.GLOBAL_SEED,
        )
        model.fit(train_features, labels[train_idx])

        predictions = model.predict(val_features)
        probabilities = model.predict_proba(val_features)[:, 1]

        accuracy = float(accuracy_score(labels[val_idx], predictions))
        try:
            auc = float(roc_auc_score(labels[val_idx], probabilities))
        except ValueError:
            auc = 0.5

        accuracies.append(accuracy)
        aucs.append(auc)

    return float(np.mean(accuracies)), float(np.mean(aucs))


def run_validation() -> None:
    """Run feature ablation validation and log results.

    Returns
    -------
    None

    """
    fs.seed_everything()
    cache_path = fs.cache_path("train", "standard", 0)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")

    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")

    h0_matrix, h1_matrix, hs_matrix, combined_matrix, geo_matrix, labels = (
        build_feature_sets(cache_path)
    )

    features_path = fs.CACHE_DIR / "ablation_features.pkl"
    fs.save_pickle(
        {
            "h0": h0_matrix,
            "h1": h1_matrix,
            "hs": hs_matrix,
            "combined": combined_matrix,
            "geo": geo_matrix,
            "labels": labels,
        },
        features_path,
    )

    print("| Condition | Dim | Mean Accuracy | Mean AUC |")
    print("| --- | --- | --- | --- |")

    conditions = [
        ("H0", h0_matrix),
        ("H1", h1_matrix),
        ("HS", hs_matrix),
        ("Combined", combined_matrix),
        ("Hu", geo_matrix),
    ]

    for name, features in conditions:
        mean_acc, mean_auc = evaluate_condition(features, labels)
        print(
            "| {name} | {dim} | {acc} | {auc} |".format(
                name=name,
                dim=fs.format_value(features.shape[1]),
                acc=fs.format_value(mean_acc),
                auc=fs.format_value(mean_auc),
            )
        )
        fs.append_perf_log(
            {
                "step": "3_signal",
                "run_id": run_id,
                "condition": name,
                "dim": int(features.shape[1]),
                "mean_acc": float(mean_acc),
                "mean_auc": float(mean_auc),
                "split": "train",
                "perturbation": "standard",
                "level": 0,
            }
        )

    print(f"\nSaved features to {features_path}")


if __name__ == "__main__":
    run_validation()
