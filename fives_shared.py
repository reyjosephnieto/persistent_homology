"""Shared configuration and helpers for the retinal TDA pipeline."""

from __future__ import annotations

import datetime
import json
import multiprocessing
import os
import pathlib
import pickle
import random
import subprocess
import sys
import typing

import cv2
import gudhi
import numpy as np

ImageArray = typing.Annotated[np.ndarray, "H,W,C"]
GreyArray = typing.Annotated[np.ndarray, "H,W"]
DiagramArray = typing.Annotated[np.ndarray, "N,2"]
FeatureVector = typing.Annotated[np.ndarray, "7"]
TopoStatVector = typing.Annotated[np.ndarray, "2"]

DATASET_ROOT = pathlib.Path(
    "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"
)
CACHE_DIR = pathlib.Path("cache_parts")
PERF_LOG_PATH = pathlib.Path("perf_log.jsonl")

GLOBAL_SEED = 42
BYTES_PER_WORKER = 2 * 1024**3
RESERVED_CORES = 1
RESERVED_MEM_BYTES = 2 * 1024**3

PROTOCOL_GROUPS: typing.Dict[str, typing.Sequence[str]] = {
    "Mechanical": ("rotation", "resolution", "blur"),
    "Radiometric": ("drift", "gamma", "contrast"),
    "Failure": ("gau_noise", "spepper_noise", "poi_noise", "bit_depth"),
}

GAMMA_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.1, 3.1, 0.1), 1)
)
CONTRAST_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.5, 3.1, 0.1), 1)
)
DRIFT_LEVELS: typing.Tuple[int, ...] = tuple(np.arange(-150, 160, 10))
ROTATION_LEVELS: typing.Tuple[int, ...] = tuple(np.arange(-10, 11, 1))
BLUR_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.0, 1.51, 0.15), 2)
)
GAU_NOISE_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.0, 0.201, 0.02), 2)
)
POI_NOISE_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.0, 0.201, 0.02), 2)
)
SPEPPER_NOISE_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.0, 0.0251, 0.0025), 4)
)
BIT_DEPTH_LEVELS: typing.Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
RESOLUTION_LEVELS: typing.Tuple[int, ...] = (
    64,
    128,
    256,
    512,
    768,
    1024,
    1280,
    1600,
    1920,
    2048,
)

PROTOCOLS: typing.Sequence[
    typing.Tuple[str, typing.Sequence[typing.Union[int, float]]]
] = (
    ("standard", (0,)),
    ("gamma", GAMMA_LEVELS),
    ("contrast", CONTRAST_LEVELS),
    ("drift", DRIFT_LEVELS),
    ("rotation", ROTATION_LEVELS),
    ("blur", BLUR_LEVELS),
    ("gau_noise", GAU_NOISE_LEVELS),
    ("poi_noise", POI_NOISE_LEVELS),
    ("spepper_noise", SPEPPER_NOISE_LEVELS),
    ("bit_depth", BIT_DEPTH_LEVELS),
    ("resolution", RESOLUTION_LEVELS),
)


def _compute_max_workers() -> int:
    """Compute worker count from CPU and memory limits.

    Returns
    -------
    int
        Worker count.
    """
    cpu_total = multiprocessing.cpu_count()
    cpu_workers = max(1, cpu_total - RESERVED_CORES)
    mem_workers = cpu_workers

    try:
        has_pages = (
            hasattr(os, "sysconf")
            and "SC_PAGE_SIZE" in os.sysconf_names
            and "SC_PHYS_PAGES" in os.sysconf_names
        )
        if has_pages:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and pages > 0:
                mem_bytes = page_size * pages
                available_bytes = max(0, mem_bytes - RESERVED_MEM_BYTES)
                mem_workers = max(
                    1, int(available_bytes // BYTES_PER_WORKER)
                )
    except (OSError, ValueError, AttributeError):
        mem_workers = cpu_workers

    return max(1, min(cpu_workers, mem_workers))


MAX_WORKERS = _compute_max_workers()


def report_resource_allocation(step_label: str) -> None:
    """Print cores and memory allocation at step start.

    Parameters
    ----------
    step_label : str
        Label for the running step.
    """
    cpu_total = multiprocessing.cpu_count()
    mem_total_gb = 0.0
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and pages > 0:
                mem_total_gb = (page_size * pages) / (1024**3)
    except (OSError, ValueError, AttributeError):
        mem_total_gb = 0.0

    print("\nRunning {label}...".format(label=step_label))
    if mem_total_gb > 0:
        print(
            "RESOURCE ALLOCATION: {workers}/{total} cores, {mem:.2f} GB memory".format(
                workers=MAX_WORKERS,
                total=cpu_total,
                mem=mem_total_gb,
            )
        )
    else:
        print(
            "RESOURCE ALLOCATION: {workers}/{total} cores".format(
                workers=MAX_WORKERS,
                total=cpu_total,
            )
        )


def make_rng(*tokens: typing.Any) -> np.random.Generator:
    """Create a seeded NumPy generator.

    Parameters
    ----------
    *tokens : typing.Any
        Stable tokens used to derive the seed.

    Returns
    -------
    numpy.random.Generator
        Generator seeded from GLOBAL_SEED and tokens.
    """
    if tokens:
        payload = "|".join(str(token) for token in tokens).encode("utf-8")
        digest = np.frombuffer(payload, dtype=np.uint8).sum()
        seed = int(digest) ^ GLOBAL_SEED
        return np.random.default_rng(seed)
    return np.random.default_rng(GLOBAL_SEED)


def seed_everything(seed: typing.Optional[int] = None) -> None:
    """Seed Python and NumPy RNGs for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed to use. Defaults to GLOBAL_SEED when omitted.
    """
    seed_value = GLOBAL_SEED if seed is None else int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)


def format_level(level: typing.Union[int, float]) -> str:
    """Format a perturbation level for cache keys.

    Parameters
    ----------
    level : int or float
        Perturbation magnitude.

    Returns
    -------
    str
        Canonical string.
    """
    return format(level, "g")


def format_value(value: typing.Any) -> typing.Any:
    """Format numeric values to three decimals.

    Parameters
    ----------
    value : typing.Any
        Value to format.

    Returns
    -------
    typing.Any
        Formatted string for numeric values, original otherwise.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.3f}"
    return value


def cache_path(
    split: str,
    perturbation_type: str,
    level: typing.Union[int, float],
) -> pathlib.Path:
    """Build the cache file path for a split and perturbation.

    Parameters
    ----------
    split : str
        Dataset split name.
    perturbation_type : str
        Perturbation family name.
    level : int or float
        Perturbation magnitude.

    Returns
    -------
    pathlib.Path
        Cache file path.
    """
    level_text = format_level(level)
    filename = f"{split}_{perturbation_type}_{level_text}.pkl"
    return CACHE_DIR / filename


def read_cache_stream(path: pathlib.Path) -> typing.List[typing.Any]:
    """Load cache items from a pickle stream.

    Parameters
    ----------
    path : pathlib.Path
        Cache stream path.

    Returns
    -------
    list
        Cache items in stream order.
    """
    items: typing.List[typing.Any] = []
    with path.open("rb") as handle:
        while True:
            try:
                item = pickle.load(handle)
            except EOFError:
                break
            items.append(item)
    return items


def save_pickle(obj: typing.Any, path: pathlib.Path) -> None:
    """Write an object to a pickle file.

    Parameters
    ----------
    obj : typing.Any
        Object to serialize.
    path : pathlib.Path
        Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: pathlib.Path) -> typing.Any:
    """Read an object from a pickle file.

    Parameters
    ----------
    path : pathlib.Path
        Input path.

    Returns
    -------
    typing.Any
        Deserialized object.
    """
    with path.open("rb") as handle:
        return pickle.load(handle)


def append_perf_log(record: typing.Dict[str, typing.Any]) -> None:
    """Append a JSON record to perf_log.jsonl.

    Parameters
    ----------
    record : dict
        JSON-serializable record.
    """
    payload = {key: format_value(value) for key, value in record.items()}
    payload.setdefault(
        "timestamp", datetime.datetime.utcnow().isoformat() + "Z"
    )
    with PERF_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        handle.write("\n")


def run_step(path: pathlib.Path, label: str) -> None:
    """Run a pipeline step via the current Python executable.

    Parameters
    ----------
    path : pathlib.Path
        Script path to execute.
    label : str
        Label for logging.
    """
    print("\n=== {label} ===".format(label=label))
    subprocess.run([sys.executable, str(path)], check=True, cwd=path.parent)


def load_image(path: pathlib.Path) -> GreyArray:
    """Load an image and return the green channel as uint8.

    Parameters
    ----------
    path : pathlib.Path
        Image file path.

    Returns
    -------
    numpy.ndarray
        Green-channel image.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    if image.ndim == 2:
        green = image
    else:
        green = image[:, :, 1]
    if green.dtype != np.uint8:
        green = green.astype(np.uint8)
    return green


def preprocess_green_clahe(img: ImageArray) -> GreyArray:
    """Apply CLAHE to the green channel.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        CLAHE-processed green channel.
    """
    if img.ndim == 3:
        green = img[:, :, 1]
    else:
        green = img

    if green.dtype != np.uint8:
        green = green.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    return enhanced


def compute_cubical_persistence_preprocessed(
    img: GreyArray,
) -> typing.Tuple[DiagramArray, DiagramArray, DiagramArray]:
    """Return sublevel H0/H1 and superlevel H0 (HS) diagrams.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed 2D image.

    Returns
    -------
    tuple of numpy.ndarray
        (d0_sub, d1_sub, d0_sup) diagrams.
    """
    cells = np.ascontiguousarray(img.astype(np.float32))
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=cells)
    complex_.persistence(homology_coeff_field=2)

    d0_sub = complex_.persistence_intervals_in_dimension(0)
    d1_sub = complex_.persistence_intervals_in_dimension(1)

    img_inv = 255 - img
    inv_cells = np.ascontiguousarray(img_inv.astype(np.float32))
    inv_complex = gudhi.CubicalComplex(top_dimensional_cells=inv_cells)
    inv_complex.persistence(homology_coeff_field=2)
    d0_sup = inv_complex.persistence_intervals_in_dimension(0)

    d0_sub_array = np.asarray(d0_sub, dtype=np.float32).reshape(-1, 2)
    d1_sub_array = np.asarray(d1_sub, dtype=np.float32).reshape(-1, 2)
    d0_sup_array = np.asarray(d0_sup, dtype=np.float32).reshape(-1, 2)

    return d0_sub_array, d1_sub_array, d0_sup_array


def compute_cubical_persistence(
    img: ImageArray,
) -> typing.Tuple[DiagramArray, DiagramArray, DiagramArray]:
    """Compute sublevel H0/H1 and superlevel H0 (HS) with CLAHE preprocessing.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.

    Returns
    -------
    tuple of numpy.ndarray
        (d0_sub, d1_sub, d0_sup) diagrams.
    """
    preprocessed = preprocess_green_clahe(img)
    return compute_cubical_persistence_preprocessed(preprocessed)


def extract_topological_features(diagram: DiagramArray) -> TopoStatVector:
    """Return top-5 sum and total persistence.

    Parameters
    ----------
    diagram : numpy.ndarray
        Persistence diagram.

    Returns
    -------
    numpy.ndarray
        Feature vector [top_5_sum, total_persistence].
    """
    if diagram.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    finite = diagram[np.isfinite(diagram).all(axis=1)]
    if finite.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = np.maximum(lifetimes, 0.0)
    if lifetimes.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    total_persistence = float(lifetimes.sum())
    sorted_lifetimes = np.sort(lifetimes)[::-1]
    top_5_sum = float(sorted_lifetimes[:5].sum())

    return np.array([top_5_sum, total_persistence], dtype=np.float32)


def get_geometric_features_preprocessed(img: GreyArray) -> FeatureVector:
    """Return log-modulus Hu moments from a preprocessed image.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed 2D image.

    Returns
    -------
    numpy.ndarray
        Seven Hu moments after log transform.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    eps = 1e-12
    log_hu = np.sign(hu) * np.log10(np.abs(hu) + eps)
    return log_hu.astype(np.float32)


def get_geometric_features(img: ImageArray) -> FeatureVector:
    """Return log-modulus Hu moments from a raw image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Seven Hu moments after log transform.
    """
    preprocessed = preprocess_green_clahe(img)
    return get_geometric_features_preprocessed(preprocessed)


def apply_perturbations(
    img: np.ndarray,
    perturbation_type: str,
    level: typing.Union[int, float],
    rng: typing.Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply a perturbation to an in-memory array.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    perturbation_type : str
        Perturbation name.
    level : int or float
        Perturbation magnitude.
    rng : numpy.random.Generator, optional
        Random generator for stochastic perturbations.

    Returns
    -------
    numpy.ndarray
        Perturbed image.
    """
    if perturbation_type == "standard":
        return img.copy()

    rng = rng or make_rng(perturbation_type, format_level(level))
    img_uint8 = img.astype(np.uint8, copy=False)

    if perturbation_type == "resolution":
        size = int(level)
        if size <= 0:
            raise ValueError("Resolution size must be positive.")
        return cv2.resize(
            img_uint8,
            (size, size),
            interpolation=cv2.INTER_AREA,
        )

    if perturbation_type == "rotation":
        angle = float(level)
        height, width = img_uint8.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            img_uint8,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    if perturbation_type == "blur":
        sigma = float(level)
        if sigma <= 0:
            return img_uint8.copy()
        return cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if perturbation_type == "gamma":
        gamma = float(level)
        if gamma <= 0:
            raise ValueError("Gamma must be positive.")
        normalised = img_uint8.astype(np.float32) / 255.0
        adjusted = np.power(normalised, gamma) * 255.0
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    if perturbation_type == "contrast":
        factor = float(level)
        adjusted = 127.5 + factor * (img_uint8.astype(np.float32) - 127.5)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    if perturbation_type == "drift":
        offset = float(level)
        adjusted = img_uint8.astype(np.float32) + offset
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    if perturbation_type == "bit_depth":
        bits = int(level)
        if bits < 1 or bits > 8:
            raise ValueError("Bit depth must be in [1, 8].")
        levels = 2**bits
        step = 256 // levels
        return (img_uint8 // step) * step

    if perturbation_type == "gau_noise":
        sigma = float(level)
        if sigma <= 0:
            return img_uint8.copy()
        sigma_dn = sigma * 255.0
        noise = rng.normal(0.0, sigma_dn, img_uint8.shape).astype(np.float32)
        noisy = img_uint8.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    if perturbation_type == "poi_noise":
        severity = float(level)
        if severity <= 0:
            return img_uint8.copy()
        if severity >= 1:
            severity = 0.99
        scale = 1.0 - severity
        sampled = rng.poisson(img_uint8.astype(np.float32) * scale).astype(
            np.float32
        )
        noisy = sampled / scale
        return np.clip(noisy, 0, 255).astype(np.uint8)

    if perturbation_type == "spepper_noise":
        severity = float(level)
        if severity <= 0:
            return img_uint8.copy()
        if severity > 1:
            raise ValueError("Salt-and-pepper severity must be in [0, 1].")
        mask = rng.random(img_uint8.shape[:2])
        salt_mask = mask < (severity / 2.0)
        pepper_mask = (mask >= (severity / 2.0)) & (mask < severity)
        noisy = img_uint8.copy()
        if img_uint8.ndim == 2:
            noisy[salt_mask] = 255
            noisy[pepper_mask] = 0
        else:
            noisy[salt_mask] = (255, 255, 255)
            noisy[pepper_mask] = (0, 0, 0)
        return noisy

    raise ValueError(f"Unknown perturbation type: {perturbation_type}")
