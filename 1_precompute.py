# 1_precompute.py
"""Precompute caches."""

import concurrent.futures
import itertools
import pathlib
import pickle
import typing

import cv2
import fives_shared as fs

CacheItem = typing.Tuple[
    typing.Tuple[fs.DiagramArray, fs.DiagramArray, fs.DiagramArray],
    fs.FeatureVector,
    int,
]

FLUSH_EVERY = 50
BASE_RES = (1024, 1024)


def _write_cache_stream(
    buffer: typing.List[CacheItem],
    output_path: pathlib.Path,
) -> None:
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


def _process_item(
    path: pathlib.Path,
    label: int,
    perturbation_type: str,
    level: typing.Union[int, float],
) -> CacheItem:
    """Compute persistence diagrams and geometry for one image.

    Parameters
    ----------
    path : pathlib.Path
        Image path.
    label : int
        Class label.
    perturbation_type : str
        Perturbation name.
    level : int or float
        Perturbation magnitude.

    Returns
    -------
    CacheItem
        ((d0_sub, d1_sub, d0_sup), geom_vec, label).

    """
    image = fs.load_image(path)
    if perturbation_type == "resolution":
        target = int(level)
        if target <= 0:
            raise ValueError("Resolution size must be a positive integer.")
        perturbed = cv2.resize(
            image,
            (target, target),
            interpolation=cv2.INTER_AREA,
        )
    else:
        base = cv2.resize(image, BASE_RES, interpolation=cv2.INTER_AREA)
        rng = fs.make_rng(path, perturbation_type, fs.format_level(level))
        perturbed = fs.apply_perturbations(
            base,
            perturbation_type,
            level,
            rng=rng,
        )
    d0_sub, d1_sub, d0_sup = fs.compute_cubical_persistence(perturbed)
    geom_vec = fs.get_geometric_features(perturbed)
    return (d0_sub, d1_sub, d0_sup), geom_vec, label


def _run_protocol(
    split: str,
    perturbation_type: str,
    levels: typing.Sequence[typing.Union[int, float]],
    progress: typing.Dict[str, int],
) -> None:
    """Run a perturbation protocol for a split.

    Parameters
    ----------
    split : str
        Dataset split name.
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
    items = fs.list_split_paths(split)

    for level in levels:
        output_path = fs.cache_path(split, perturbation_type, level)
        if output_path.exists():
            output_path.unlink()

        buffer: typing.List[CacheItem] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=fs.MAX_WORKERS
        ) as executor:
            total = len(items)
            chunksize = max(1, total // (fs.MAX_WORKERS * 4))
            results = executor.map(
                _process_item,
                (path for path, _ in items),
                (label for _, label in items),
                itertools.repeat(perturbation_type),
                itertools.repeat(level),
                chunksize=chunksize,
            )
            display_type = (
                "standard"
                if perturbation_type == "standard"
                else perturbation_type
            )
            display_label = "{ptype} {level}".format(
                ptype=display_type,
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
    total_levels = sum(len(levels) for _, levels in fs.PROTOCOLS)
    total_items = 0
    for split in ("train", "test"):
        total_items += len(fs.list_split_paths(split, verbose=False))
    total_work = total_items * total_levels
    progress = {"completed": 0, "total": total_work}

    for split in ("train", "test"):
        for perturbation_type, levels in fs.PROTOCOLS:
            _run_protocol(split, perturbation_type, levels, progress)


if __name__ == "__main__":
    run_precompute()
