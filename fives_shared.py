# fives_shared.py
"""Pipeline helpers."""

import datetime
import hashlib
import json
import multiprocessing
import os
import pathlib
import pickle
import subprocess
import sys
import typing

_THREAD_ENV_VARS = (
    ("OMP_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
)
for key, value in _THREAD_ENV_VARS:
    os.environ.setdefault(key, value)

import cv2
import gudhi
import numpy as np

ImageArray = typing.Annotated[np.ndarray, "H,W,C"]
GreyArray = typing.Annotated[np.ndarray, "H,W"]
DiagramArray = typing.Annotated[np.ndarray, "N,2"]
FeatureVector = typing.Annotated[np.ndarray, "7"]
TopoStatVector = typing.Annotated[np.ndarray, "3"]

DATASET_ROOT = pathlib.Path(
    "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"
)
CACHE_DIR = pathlib.Path("cache_parts")
PERF_LOG_PATH = pathlib.Path("perf_log.jsonl")
BYTES_PER_WORKER = 2 * 1024**3
GLOBAL_SEED = 42
RESERVED_CORES = 1
RESERVED_MEM_BYTES = 2 * 1024**3
NOISE_SAMPLE_COUNT = 24
NOISE_CALIBRATION_RES = (1024, 1024)
NOISE_HIGH_PASS_SIGMA = 3.0
NOISE_CORRELATION_SIGMA = 1.0

GAMMA_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.5, 1.6, 0.1), 1)
)
CONTRAST_LEVELS: typing.Tuple[float, ...] = tuple(
    np.round(np.arange(0.5, 1.6, 0.1), 1)
)
DRIFT_LEVELS: typing.Tuple[int, ...] = tuple(np.arange(-50, 60, 10))
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
    """Compute a worker count from CPU and memory limits.

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
np.random.seed(GLOBAL_SEED)
_RETINA_MASK_CACHE: typing.Dict[typing.Tuple[int, int], np.ndarray] = {}
_READ_NOISE_SIGMA: typing.Optional[float] = None


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
        digest = hashlib.sha256(payload).digest()
        seed = int.from_bytes(digest[:8], "big") ^ GLOBAL_SEED
        return np.random.default_rng(seed)
    return np.random.default_rng(GLOBAL_SEED)


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


def list_split_paths(
    split: str,
    verbose: bool = True,
) -> typing.List[typing.Tuple[pathlib.Path, int]]:
    """Return a balanced list of (path, label) for a split.

    Parameters
    ----------
    split : str
        Dataset split name.
    verbose : bool, optional
        Whether to print split counts.

    Returns
    -------
    list of tuple[pathlib.Path, int]
        Image paths and labels.

    """
    split_root = DATASET_ROOT / split / "Original"
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    all_paths = sorted(split_root.glob("*.png"))
    normal_paths = [path for path in all_paths if path.name.endswith("_N.png")]
    diabetic_paths = [path for path in all_paths if path.name.endswith("_D.png")]

    if split == "train":
        if len(normal_paths) < 150 or len(diabetic_paths) < 150:
            raise ValueError(
                "Insufficient samples for train split; require 150 per class."
            )
        normal_paths = normal_paths[0:150]
        diabetic_paths = diabetic_paths[0:150]
    elif split == "test":
        remainder_size = min(len(normal_paths), len(diabetic_paths))
        normal_paths = normal_paths[:remainder_size]
        diabetic_paths = diabetic_paths[:remainder_size]
    else:
        raise ValueError("Split must be 'train' or 'test'.")

    items: typing.List[typing.Tuple[pathlib.Path, int]] = []
    for path in normal_paths:
        items.append((path, 0))
    for path in diabetic_paths:
        items.append((path, 1))

    if verbose:
        print(
            "Loaded {split}: {normal} Normal, {diabetic} Diabetic".format(
                split=split,
                normal=len(normal_paths),
                diabetic=len(diabetic_paths),
            )
        )
    return items


def load_image(path: pathlib.Path) -> ImageArray:
    """Load a BGR image without resizing.

    Parameters
    ----------
    path : pathlib.Path
        Image path.

    Returns
    -------
    numpy.ndarray
        BGR image array.

    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def _get_retina_mask(target_shape: typing.Tuple[int, int]) -> np.ndarray:
    """Return a boolean mask for the retinal field.

    Parameters
    ----------
    target_shape : tuple of int
        Target (height, width).

    Returns
    -------
    numpy.ndarray
        Boolean mask where True indicates retinal pixels.

    """
    cached = _RETINA_MASK_CACHE.get(target_shape)
    if cached is not None:
        return cached

    mask_path = pathlib.Path(__file__).resolve().parent / "mask_circle.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    if mask.ndim == 3 and mask.shape[2] == 4:
        alpha = mask[:, :, 3]
    elif mask.ndim == 3:
        alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        alpha = mask
    if alpha.shape[:2] != target_shape:
        alpha = cv2.resize(
            alpha,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    alpha = alpha.astype(np.float32) / 255.0
    retina_mask = alpha < 0.5
    _RETINA_MASK_CACHE[target_shape] = retina_mask
    return retina_mask


def _get_read_noise_sigma() -> float:
    """Estimate read noise sigma from training images.

    Returns
    -------
    float
        Estimated read noise sigma in DN.

    """
    global _READ_NOISE_SIGMA
    if _READ_NOISE_SIGMA is not None:
        return _READ_NOISE_SIGMA

    items = list_split_paths("train", verbose=False)
    if not items:
        _READ_NOISE_SIGMA = 1.0
        return _READ_NOISE_SIGMA

    mask = _get_retina_mask(NOISE_CALIBRATION_RES)
    sigmas: typing.List[float] = []

    for path, _ in items[:NOISE_SAMPLE_COUNT]:
        image = load_image(path)
        resized = cv2.resize(
            image,
            (NOISE_CALIBRATION_RES[1], NOISE_CALIBRATION_RES[0]),
            interpolation=cv2.INTER_AREA,
        )
        green = resized[:, :, 1].astype(np.float32)
        blurred = cv2.GaussianBlur(
            green,
            (0, 0),
            sigmaX=NOISE_HIGH_PASS_SIGMA,
            sigmaY=NOISE_HIGH_PASS_SIGMA,
        )
        residual = green - blurred
        residual = residual[mask]
        if residual.size == 0:
            continue
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        sigma = 1.4826 * mad
        if sigma > 0 and np.isfinite(sigma):
            sigmas.append(sigma)

    _READ_NOISE_SIGMA = float(np.median(sigmas)) if sigmas else 1.0
    return _READ_NOISE_SIGMA


def _scale_noise_sigma(
    level: float,
    levels: typing.Sequence[typing.Union[int, float]],
    base_sigma: float,
) -> float:
    """Scale a noise level to a sigma in DN.

    Parameters
    ----------
    level : float
        Noise level parameter.
    levels : sequence of int or float
        Allowed levels for the protocol.
    base_sigma : float
        Base sigma estimate in DN.

    Returns
    -------
    float
        Sigma in DN.

    """
    max_level = float(max(levels)) if levels else 0.0
    if max_level <= 0:
        return 0.0
    return (level / max_level) * base_sigma


def _normalize_noise(
    noise: np.ndarray,
    target_sigma: float,
    mask: typing.Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rescale noise to a target sigma.

    Parameters
    ----------
    noise : numpy.ndarray
        Noise array.
    target_sigma : float
        Target standard deviation.
    mask : numpy.ndarray, optional
        Boolean mask for sigma estimation.

    Returns
    -------
    numpy.ndarray
        Rescaled noise array.

    """
    if target_sigma <= 0:
        return noise
    if mask is None:
        sample = noise.reshape(-1)
    else:
        sample = noise[mask]
    current = float(np.std(sample)) if sample.size else 0.0
    if current > 0:
        noise = noise * (target_sigma / current)
    return noise


def apply_perturbations(
    img: ImageArray,
    perturbation_type: str,
    level: typing.Union[int, float],
    rng: typing.Optional[np.random.Generator] = None,
) -> ImageArray:
    """Apply a perturbation and keep uint8 output.

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
    ptype = perturbation_type

    if ptype == "standard":
        return img.copy()

    rng = rng or make_rng(ptype, format_level(level))

    if ptype == "resolution":
        size = int(level)
        if size <= 0:
            raise ValueError("Resolution size must be a positive integer.")
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return resized
    if ptype == "rotation":
        angle = float(level)
        height, width = img.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return rotated
    if ptype == "blur":
        sigma = float(level)
        if sigma <= 0:
            return img.copy()
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return blurred
    if ptype == "gau_noise":
        sigma = float(level)
        if sigma <= 0:
            return img.copy()
        height, width = img.shape[:2]
        retina_mask = _get_retina_mask((height, width))
        base_sigma = _get_read_noise_sigma()
        sigma_dn = _scale_noise_sigma(sigma, GAU_NOISE_LEVELS, base_sigma)
        if sigma_dn <= 0:
            return img.copy()
        noise = rng.normal(0.0, sigma_dn, img.shape).astype(np.float32)
        noise = cv2.GaussianBlur(
            noise,
            (0, 0),
            sigmaX=NOISE_CORRELATION_SIGMA,
            sigmaY=NOISE_CORRELATION_SIGMA,
        )
        noise = _normalize_noise(noise, sigma_dn, retina_mask)
        if img.ndim == 2:
            noise *= retina_mask
        else:
            noise *= retina_mask[:, :, None]
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if ptype == "poi_noise":
        severity = float(level)
        if severity <= 0:
            return img.copy()
        if severity >= 1:
            severity = 0.99
        height, width = img.shape[:2]
        retina_mask = _get_retina_mask((height, width))
        scale = 1.0 - severity
        simulated = rng.poisson(img.astype(np.float32) * scale).astype(
            np.float32
        )
        noisy = simulated / scale
        base_sigma = _get_read_noise_sigma()
        read_sigma = _scale_noise_sigma(
            severity, POI_NOISE_LEVELS, base_sigma
        )
        read_noise = rng.normal(0.0, read_sigma, img.shape).astype(np.float32)
        read_noise = cv2.GaussianBlur(
            read_noise,
            (0, 0),
            sigmaX=NOISE_CORRELATION_SIGMA,
            sigmaY=NOISE_CORRELATION_SIGMA,
        )
        read_noise = _normalize_noise(read_noise, read_sigma, retina_mask)
        noisy = noisy + read_noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        output = img.copy()
        output[retina_mask] = noisy[retina_mask]
        return output
    if ptype == "spepper_noise":
        severity = float(level)
        if severity <= 0:
            return img.copy()
        if severity > 1:
            raise ValueError("Salt-and-pepper severity must be in [0, 1].")
        height, width = img.shape[:2]
        retina_mask = _get_retina_mask((height, width))
        mask = rng.random((height, width))
        salt_mask = (mask < (severity / 2.0)) & retina_mask
        pepper_mask = (
            (mask >= (severity / 2.0)) & (mask < severity) & retina_mask
        )
        noisy = img.copy()
        if img.ndim == 2:
            noisy[salt_mask] = 255
            noisy[pepper_mask] = 0
        else:
            noisy[salt_mask] = (255, 255, 255)
            noisy[pepper_mask] = (0, 0, 0)
        return noisy
    if ptype == "bit_depth":
        bits = int(level)
        if bits < 1 or bits > 8:
            raise ValueError("Bit depth must be an integer in [1, 8].")
        levels = 2**bits
        step = 256 // levels
        quantised = (img // step) * step
        return quantised.astype(np.uint8)
    if ptype == "drift":
        offset = int(level)
        adjusted = np.clip(
            img.astype(np.int16) + offset,
            0,
            255,
        ).astype(np.uint8)
        return adjusted

    img_float = img.astype(np.float32)

    if ptype == "gamma":
        gamma = float(level)
        if gamma <= 0:
            raise ValueError("Gamma must be positive.")
        normalised = img_float / 255.0
        adjusted = np.power(normalised, gamma) * 255.0
    elif ptype == "contrast":
        factor = float(level)
        adjusted = 127.5 + factor * (img_float - 127.5)
    else:
        raise ValueError(f"Unknown perturbation type: {ptype}")

    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


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


def compute_cubical_persistence(
    img: ImageArray,
) -> typing.Tuple[DiagramArray, DiagramArray, DiagramArray]:
    """Return sublevel H0/H1 and superlevel H0 diagrams.

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
    cells = np.ascontiguousarray(preprocessed.astype(np.float32))
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=cells)
    complex_.persistence(homology_coeff_field=2, min_persistence=4.0)

    d0_sub = complex_.persistence_intervals_in_dimension(0)
    d1_sub = complex_.persistence_intervals_in_dimension(1)

    img_inv = 255 - preprocessed
    inv_cells = np.ascontiguousarray(img_inv.astype(np.float32))
    inv_complex = gudhi.CubicalComplex(top_dimensional_cells=inv_cells)
    inv_complex.persistence(homology_coeff_field=2, min_persistence=4.0)
    d0_sup = inv_complex.persistence_intervals_in_dimension(0)

    d0_sub_array = np.asarray(d0_sub, dtype=np.float32).reshape(-1, 2)
    d1_sub_array = np.asarray(d1_sub, dtype=np.float32).reshape(-1, 2)
    d0_sup_array = np.asarray(d0_sup, dtype=np.float32).reshape(-1, 2)

    return d0_sub_array, d1_sub_array, d0_sup_array


def extract_topological_features(diagram: DiagramArray) -> TopoStatVector:
    """Return count, max persistence, and total persistence.

    Parameters
    ----------
    diagram : numpy.ndarray
        Persistence diagram.

    Returns
    -------
    numpy.ndarray
        Feature vector [count, max_persistence, total_persistence].

    """
    if diagram.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    finite = diagram[np.isfinite(diagram).all(axis=1)]
    if finite.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = np.maximum(lifetimes, 0.0)

    count = float(finite.shape[0])
    max_persistence = float(lifetimes.max()) if lifetimes.size else 0.0
    total_persistence = float(lifetimes.sum()) if lifetimes.size else 0.0

    return np.array(
        [count, max_persistence, total_persistence], dtype=np.float32
    )


def get_geometric_features(img: ImageArray) -> FeatureVector:
    """Return log-modulus Hu moments.

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
    moments = cv2.moments(preprocessed)
    hu = cv2.HuMoments(moments).flatten()
    eps = 1e-12
    log_hu = np.sign(hu) * np.log10(np.abs(hu) + eps)

    return log_hu.astype(np.float32)


def save_pickle(obj: typing.Any, path: pathlib.Path) -> None:
    """Write an object to a pickle file.

    Parameters
    ----------
    obj : typing.Any
        Object to serialize.
    path : pathlib.Path
        Output path.

    Returns
    -------
    None

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

    Returns
    -------
    None

    """
    payload = {key: format_value(value) for key, value in record.items()}
    payload.setdefault(
        "timestamp", datetime.datetime.utcnow().isoformat() + "Z"
    )
    with PERF_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        handle.write("\n")


def run_step(script_path: pathlib.Path, label: str) -> None:
    """Run a pipeline step as a subprocess.

    Parameters
    ----------
    script_path : pathlib.Path
        Script to execute.
    label : str
        Human-readable step label.

    Returns
    -------
    None

    """
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n=== Running {label}: {script_path.name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)
