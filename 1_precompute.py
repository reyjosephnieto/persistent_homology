# 1_precompute.py
"""Precompute caches from staged in-memory tensors."""

from __future__ import annotations

import concurrent.futures
import itertools
import pathlib
import pickle
from collections.abc import Sequence

import numpy as np

import fives_shared as fs

CacheItem = tuple[
    tuple[fs.DiagramArray, fs.DiagramArray, fs.DiagramArray],
    fs.FeatureVector,
    int,
]

DATA_DIR = pathlib.Path("data")
CLEAN_IMAGES_PATH = DATA_DIR / "clean_images.npy"
CLEAN_LABELS_PATH = DATA_DIR / "clean_labels.npy"
RAW_IMAGES_PATH = DATA_DIR / "raw_images.npy"
RAW_PATHS_PATH = DATA_DIR / "raw_paths.npy"
TRAIN_INDICES_PATH = DATA_DIR / "train_indices.npy"
TEST_INDICES_PATH = DATA_DIR / "test_indices.npy"

FLUSH_EVERY = 50

_CLEAN_IMAGES: np.ndarray | None = None
_RAW_IMAGES: np.ndarray | None = None
_RAW_PATHS: np.ndarray | None = None
_LABELS: np.ndarray | None = None


def _load_array(
    path: pathlib.Path,
    allow_pickle: bool = False,
) -> np.ndarray:
    """Load a NumPy array with memory mapping when possible.

    Parameters
    ----------
    path : pathlib.Path
        Input .npy path.
    allow_pickle : bool, optional
        Whether to allow pickled object arrays.

    Returns
    -------
    numpy.ndarray
        Loaded array.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing array: {path}")

    if allow_pickle:
        return np.load(path, allow_pickle=True)

    return np.load(path, mmap_mode="r")


def _init_worker() -> None:
    """Initialize worker with shared arrays.

    Returns
    -------
    None
    """
    global _CLEAN_IMAGES, _RAW_IMAGES, _RAW_PATHS, _LABELS

    _CLEAN_IMAGES = _load_array(CLEAN_IMAGES_PATH)
    _LABELS = _load_array(CLEAN_LABELS_PATH)

    if RAW_IMAGES_PATH.exists():
        raw_images = _load_array(RAW_IMAGES_PATH, allow_pickle=True)
        if raw_images.dtype == object:
            _RAW_IMAGES = None
        else:
            _RAW_IMAGES = raw_images
    if RAW_PATHS_PATH.exists():
        _RAW_PATHS = _load_array(RAW_PATHS_PATH, allow_pickle=True)


def _write_cache_stream(buffer: list[CacheItem], output_path: pathlib.Path) -> None:
    """Append buffered cache items to a pickle stream.

    Parameters
    ----------
    buffer : list of CacheItem
        Items to write.
    output_path : pathlib.Path
        Output pickle stream path.

    Returns
    -------
    None
    """
    if not buffer:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("ab") as handle:
        for item in buffer:
            pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _get_image(index: int, perturbation_type: str) -> np.ndarray:
    """Return the image for a given index and protocol.

    Parameters
    ----------
    index : int
        Image index.
    perturbation_type : str
        Perturbation protocol name.

    Returns
    -------
    numpy.ndarray
        Image array.
    """
    if perturbation_type == "resolution":
        if _RAW_IMAGES is not None:
            return _RAW_IMAGES[index]
        if _RAW_PATHS is not None:
            return fs.load_image(pathlib.Path(_RAW_PATHS[index]))
        raise ValueError("Missing raw_images.npy and raw_paths.npy")

    if _CLEAN_IMAGES is None:
        raise ValueError("Clean images not loaded.")
    return _CLEAN_IMAGES[index]


def _process_item(
    index: int,
    perturbation_type: str,
    level: int | float,
) -> CacheItem:
    """Compute persistence and geometry for one image.

    Parameters
    ----------
    index : int
        Image index.
    perturbation_type : str
        Perturbation protocol name.
    level : int or float
        Perturbation magnitude.

    Returns
    -------
    CacheItem
        ((d0_sub, d1_sub, d0_sup), geom_vec, label).
    """
    if _LABELS is None:
        raise ValueError("Labels not loaded.")

    image = _get_image(index, perturbation_type)
    label = int(_LABELS[index])

    if perturbation_type == "resolution":
        perturbed = fs.apply_perturbations(image, perturbation_type, level)
        d0_sub, d1_sub, d0_sup = fs.compute_cubical_persistence(perturbed)
        geom_vec = fs.get_geometric_features(perturbed)
    else:
        rng = fs.make_rng(index, perturbation_type, fs.format_level(level))
        perturbed = fs.apply_perturbations(
            image,
            perturbation_type,
            level,
            rng=rng,
        )
        d0_sub, d1_sub, d0_sup = fs.compute_cubical_persistence_preprocessed(
            perturbed
        )
        geom_vec = fs.get_geometric_features_preprocessed(perturbed)

    return (d0_sub, d1_sub, d0_sup), geom_vec, label


def _load_indices(path: pathlib.Path) -> np.ndarray:
    """Load an index array.

    Parameters
    ----------
    path : pathlib.Path
        Index path.

    Returns
    -------
    numpy.ndarray
        Index array.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing index file: {path}")
    return np.load(path)


def _run_protocol(
    split: str,
    indices: np.ndarray,
    perturbation_type: str,
    levels: Sequence[int | float],
    progress: dict[str, int],
) -> None:
    """Run a perturbation protocol for a split.

    Parameters
    ----------
    split : str
        Dataset split name.
    indices : numpy.ndarray
        Image indices for the split.
    perturbation_type : str
        Perturbation name.
    levels : sequence of int or float
        Perturbation magnitudes.
    progress : dict of str to int
        Global progress tracker with "completed" and "total" keys.

    Returns
    -------
    None
    """
    for level in levels:
        output_path = fs.cache_path(split, perturbation_type, level)
        if output_path.exists():
            print(
                "[{split}] {ptype} {level} exists; skipping.".format(
                    split=split,
                    ptype=perturbation_type,
                    level=fs.format_level(level),
                )
            )
            continue

        buffer: list[CacheItem] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=fs.MAX_WORKERS,
            initializer=_init_worker,
        ) as executor:
            total = int(indices.shape[0])
            chunksize = max(1, total // (fs.MAX_WORKERS * 4))
            results = executor.map(
                _process_item,
                indices,
                itertools.repeat(perturbation_type),
                itertools.repeat(level),
                chunksize=chunksize,
            )
            display_label = "{ptype} {level}".format(
                ptype=perturbation_type,
                level=fs.format_level(level),
            )
            last_percent = -1
            for index, result in enumerate(results, start=1):
                buffer.append(result)
                progress["completed"] += 1
                percent = int(100.0 * index / total) if total else 100
                if percent != last_percent:
                    overall = "{done}/{total}".format(
                        done=progress["completed"],
                        total=progress["total"],
                    )
                    status = (
                        "\r[{split}] {label} {index}/{total} "
                        "({percent:3d}%) | overall {overall}".format(
                            split=split,
                            label=display_label,
                            index=index,
                            total=total,
                            percent=percent,
                            overall=overall,
                        )
                    )
                    print(status.ljust(80), end="", flush=True)
                    last_percent = percent
                if len(buffer) >= FLUSH_EVERY:
                    _write_cache_stream(buffer, output_path)
                    buffer.clear()

        _write_cache_stream(buffer, output_path)
        print()


def run_precompute() -> None:
    """Precompute caches for all splits and protocols.

    Returns
    -------
    None
    """
    fs.report_resource_allocation("Precompute")

    train_indices = _load_indices(TRAIN_INDICES_PATH)
    test_indices = _load_indices(TEST_INDICES_PATH)

    total_levels = sum(len(levels) for _, levels in fs.PROTOCOLS)
    total_work = (len(train_indices) + len(test_indices)) * total_levels
    progress = {"completed": 0, "total": total_work}

    for split, indices in ("train", train_indices), ("test", test_indices):
        for perturbation_type, levels in fs.PROTOCOLS:
            _run_protocol(split, indices, perturbation_type, levels, progress)


if __name__ == "__main__":
    run_precompute()
