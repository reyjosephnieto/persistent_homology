# 4_generalise.py
"""Train/test generalisation check."""

import datetime
import typing

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import fives_shared as fs

DiagramArray = fs.DiagramArray

TopoVector = typing.Annotated[np.ndarray, "6"]
TopoMatrix = typing.Annotated[np.ndarray, "N,6"]
LabelArray = typing.Annotated[np.ndarray, "N"]

CacheItem = typing.Tuple[
    typing.Tuple[DiagramArray, DiagramArray, DiagramArray], np.ndarray, int
]


def load_split(split: str) -> typing.Tuple[TopoMatrix, LabelArray]:
    """Load a split and build the combined topological features.

    Parameters
    ----------
    split : str
        Dataset split name.

    Returns
    -------
    tuple of numpy.ndarray
        Topological feature matrix and labels.

    """
    path = fs.cache_path(split, "standard", 0)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache: {path}")

    items = typing.cast(typing.List[CacheItem], fs.read_cache_stream(path))
    if not items:
        raise ValueError("Empty cache stream.")

    features: typing.List[TopoVector] = []
    labels: typing.List[int] = []

    for diagrams, _, label in items:
        d0_sub, d1_sub, *_ = diagrams
        d0_array = np.asarray(d0_sub, dtype=np.float32).reshape(-1, 2)
        d1_array = np.asarray(d1_sub, dtype=np.float32).reshape(-1, 2)

        h0_vec = fs.extract_topological_features(d0_array)
        h1_vec = fs.extract_topological_features(d1_array)
        topo_vec = np.concatenate((h0_vec, h1_vec)).astype(np.float32)

        features.append(topo_vec)
        labels.append(int(label))

    feature_matrix = np.vstack(features).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    return feature_matrix, label_array


def run_failure_analysis() -> None:
    """Run the generalisation check and log metrics.

    Returns
    -------
    None

    """
    print("Loading data...")
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")

    print(
        "Train Size: {train} | Test Size: {test}".format(
            train=fs.format_value(len(y_train)),
            test=fs.format_value(len(y_test)),
        )
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_test_s = scaler.transform(x_test).astype(np.float32)

    model = LogisticRegression(
        class_weight="balanced",
        random_state=fs.GLOBAL_SEED,
    )
    model.fit(x_train_s, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(x_train_s)))
    test_acc = float(accuracy_score(y_test, model.predict(x_test_s)))
    test_auc = float(roc_auc_score(y_test, model.predict_proba(x_test_s)[:, 1]))

    print("\n" + "=" * 40)
    print("Generalisation Check")
    print("=" * 40)
    print(f"Train Accuracy: {fs.format_value(train_acc)}")
    print(f"Test Accuracy:  {fs.format_value(test_acc)}")
    print(f"Test AUC:       {fs.format_value(test_auc)}")
    print("=" * 40)
    print("Note: train/test distributions differ; use k-fold for evaluation.")

    fs.append_perf_log(
        {
            "step": "4_generalise",
            "run_id": run_id,
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "train_split": "train",
            "test_split": "test",
            "perturbation": "standard",
            "level": 0,
        }
    )


if __name__ == "__main__":
    run_failure_analysis()
