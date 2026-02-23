# Computational Stability of Cubical Homology
## Orthogonal Controls for Fault Isolation in Retinal Diagnostics

This repository contains an end-to-end pipeline for testing how stable Topological Data Analysis (TDA) features are on discrete sensor grids.

The clinical testbed is Diabetic Retinopathy screening from fundus images. For each image, the pipeline extracts the green channel (highest vessel contrast), computes cubical persistent homology, and compares:
- a **6D topological summary**: $\textbf{H}_0 \oplus \textbf{H}_1 \oplus \textbf{H}_S$
- a **7D geometric baseline**: Hu invariant moments

The comparison is run across **169 perturbed settings** (**170 total settings** including baseline) to test the **Orthogonality Hypothesis**:
- geometric invariants are more robust to affine/mechanical distortions
- topological invariants are more robust to illumination/quantisation shifts

## Requirements
- Python 3.9+
- Packages: `numpy`, `scipy`, `scikit-learn`, `opencv-python`, `gudhi`, `matplotlib`

```bash
pip install numpy scipy scikit-learn opencv-python gudhi matplotlib
```

## Dataset Setup
1. [Download the FIVES dataset (Fundus Image Vessel Segmentation)](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169?file=34969398).
2. Extract the archive directly into the workspace root.
3. The target directory must be named exactly:
   `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/`

The ingest script filters out AMD and Glaucoma partitions to isolate the balanced Normal/DR cohort ($N=400$).

## Feature Definitions
The feature design uses a coarse vectorisation strategy to preserve continuity under bottleneck-distance perturbations.

Homology notation:
- $\textbf{H}_0$: zeroth homology on the **sublevel filtration** (connected dark components)
- $\textbf{H}_1$: first homology on the **sublevel filtration** (loops/holes, e.g. vascular rings)
- $\textbf{H}_S$: zeroth homology on the **superlevel filtration** (connected bright components, e.g. exudate-like regions)
  - We write this as $\textbf{H}_S$ because $\textbf{H}_0$ is already used for sublevel zeroth homology; the `S` avoids a notation clash.

Feature vectors:
- **Topological vector (6D):** $\textbf{H}_0 \oplus \textbf{H}_1 \oplus \textbf{H}_S$, where each block is encoded as `[Top-5 Persistence Sum, Total Persistence]`
- **Geometric control (7D):** log-transformed Hu invariant moments
- **Audit vector (13D):** $\textbf{H}_0 \oplus \textbf{H}_1 \oplus \textbf{H}_S \oplus \text{Hu}$

Betti counts are intentionally excluded to avoid discontinuous count jumps under small perturbations.

## Pipeline Overview (0_ to 7_)
| Step | Script | Purpose |
| :---: | :--- | :--- |
| 0 | `0_ingest.py` | Stage dataset arrays (green channel, CLAHE branch, split indices). |
| 1 | `1_precompute.py` | Build persistence/Hu caches for every protocol level and split. |
| 2 | `2_audit.py` | Baseline statistical audit (Welch's t-test, Cohen's d, lifetimes). |
| 3 | `3_signal.py` | Baseline 5-fold CV signal check for `H0`, `H1`, `HS`, `H0H1HS`, `Hu`. |
| 4 | `4_generalise.py` | Train on standard train split, evaluate on standard test split. |
| 5 | `5_ablate.py` | Main stress audit across all perturbation protocols and levels. |
| 6 | `6_plot.py` | Render robustness panels and orthogonality plots. |
| 7 | `7_orchestrate.py` | Run steps `1 -> 6` in sequence. |

Step I/O details:
- `0_ingest.py`
  Input: `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/`
  Output: `data/clean_cohort.npz`, `data/clean_images.npy`, `data/clean_labels.npy`, `data/raw_images.npy`, `data/raw_paths.npy`, `data/train_indices.npy`, `data/test_indices.npy`
- `1_precompute.py`
  Input: `data/*.npy`, protocol ranges from `fives_shared.py`
  Output: `cache_parts/{split}_{protocol}_{level}.pkl`
- `2_audit.py`
  Input: `cache_parts/train_standard_0.pkl`
  Output: console tables and `perf_log.jsonl` entries (`2_audit`, `2_audit_lifetimes`)
- `3_signal.py`
  Input: `cache_parts/train_standard_0.pkl`
  Output: `cache_parts/ablation_features.pkl` and `perf_log.jsonl` entries (`3_signal`)
- `4_generalise.py`
  Input: `cache_parts/train_standard_0.pkl`, `cache_parts/test_standard_0.pkl`
  Output: `perf_log.jsonl` entries (`4_generalise`)
- `5_ablate.py`
  Input: baseline and perturbed caches from step 1
  Output: `results_mechanical.csv`, `results_radiometric.csv`, `results_failure.csv`, `stress_test_results_*.csv`, and `perf_log.jsonl` entries (`5_ablate`)
- `6_plot.py`
  Input: `perf_log.jsonl`
  Output: `plot_results/*.png`
- `7_orchestrate.py`
  Input: staged data and scripts
  Output: end-to-end artefacts from steps `1 -> 6`

## Running the Pipeline
Full run (recommended):

```bash
python 0_ingest.py
python 7_orchestrate.py
```

Notes:
- `7_orchestrate.py` does **not** run `0_ingest.py`; ingestion is a one-time staging step.
- With current protocol ranges in `fives_shared.py`, step 5 evaluates **169 perturbed settings** plus baseline.

Manual run (step-by-step):

```bash
python 1_precompute.py
python 2_audit.py
python 3_signal.py
python 4_generalise.py
python 5_ablate.py
python 6_plot.py
```

## Reproducibility, Safety, and Execution Notes
- **Read-only source data:** The FIVES dataset directory is treated as input-only. The pipeline reads from `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/` and writes artefacts to `data/`, `cache_parts/`, `plot_results/`, `results_*.csv`, and `perf_log.jsonl`.
- **Deterministic seed policy:** The codebase uses a global seed (`GLOBAL_SEED = 42` in `fives_shared.py`). Stochastic perturbations are generated from deterministic RNGs derived from this seed (`make_rng(...)`), and cross-validation splits use fixed `random_state` values.
- **Stable processing order:** Cache items are emitted and consumed in a fixed sequence (ordered indices and sequential pickle streaming), which keeps downstream feature alignment consistent across reruns.
- **Rerun behavior:** `1_precompute.py` skips cache files that already exist. To force full cache regeneration, clear `cache_parts/` first. `perf_log.jsonl` is append-only by default; remove it for a fresh run history.
- **Single-entry execution:** `7_orchestrate.py` is the driver for steps `1 -> 6` and re-runs the full downstream analysis pipeline from staged inputs.
- **Compute cost:** The precompute stage is the dominant bottleneck and can take multiple hours depending on hardware and worker count (`MAX_WORKERS` is chosen from CPU/memory constraints).
- **Code availability:** Public implementation and scripts are available at <https://github.com/reyjosephnieto/persistent_homology/>.

## Supplementary Notes (Condensed)
- **Why pooled cross-validation?** A quick train/test check on the official split showed severe drift (train accuracy $84.0\%$, test accuracy $41.0\%$), so stress evaluation is reported on pooled Normal/DR samples under stratified 5-fold CV.
- **Why truncation $\tau = 5.0$?** Lifetime distributions for `H0/H1/HS` cluster near mean persistence $\approx 4.0$, so $\tau = 5.0$ removes low-amplitude topological dust.

Lifetime summary (train baseline):

| **Homology** | **Mean** | **Median** | **Mode** | **Count** |
| :--- | ---: | ---: | ---: | ---: |
| $\textbf{H}_0$ | 4.080 | 3.412 | 1.482 | 8,762,399 |
| $\textbf{H}_1$ | 3.852 | 3.383 | 1.477 | 11,609,642 |
| $\textbf{H}_S$ | 4.010 | 3.422 | 1.484 | 8,668,531 |

Full univariate feature ranking by absolute effect size $|d|$ (train baseline):

| **Feature Name** | **Healthy Mean** | **Diabetic Mean** | **p-value** | **Cohen's $d$** |
| :--- | ---: | ---: | ---: | ---: |
| HS Top5 | 363.807 | 609.067 | 0.000 | 1.573 |
| H0 Top5 | 231.300 | 543.800 | 0.000 | 1.415 |
| Hu 5 | 7.680 | -1.920 | 0.000 | -0.901 |
| H1 Top5 | 466.493 | 547.920 | 0.000 | 0.879 |
| H0 Total | 127244.039 | 111088.070 | 0.000 | -0.698 |
| HS Total | 122674.039 | 109061.906 | 0.000 | -0.641 |
| H1 Total | 158138.562 | 139959.438 | 0.000 | -0.580 |
| Hu 4 | -9.832 | -9.995 | 0.004 | -0.336 |
| Hu 7 | -0.800 | -3.840 | 0.025 | -0.260 |
| Hu 3 | -11.274 | -11.196 | 0.116 | 0.182 |
| Hu 2 | -8.490 | -8.590 | 0.165 | -0.161 |
| Hu 1 | -2.634 | -2.642 | 0.386 | -0.100 |
| Hu 6 | -2.878 | -2.078 | 0.557 | 0.068 |

## Sensitivity Audit Configuration (`fives_shared.py`)
The pipeline evaluates three operational failure regimes. Global random seeds dictate stochastic noise generation (`seed_everything()`).

- **Mechanical (Aim):** Probing spatial interpolation penalties ($\lambda \times L_\mcI$).
  - Rotation: $-10^\circ \to 10^\circ$ (step 1)
  - Blur: $\sigma \in [0.0, 1.5]$ (step 0.15)
  - Resolution: $64\text{px} \to 2048\text{px}$
- **Radiometric (Shoot):** Probing sensor transfer function deviations.
  - Gamma: $0.1 \to 3.0$ (step 0.1)
  - Contrast: $0.5 \to 3.0$ (step 0.1)
  - Drift: $-150 \to 150$ (step 10)
- **Stochastic/Digital Failure:** Probing noise floors ($\eta$).
  - Gaussian Noise: $\sigma^2 \in [0.0, 0.20]$ (step 0.02)
  - Poisson Noise: $\lambda \in [0.0, 0.20]$ (step 0.02)
  - Salt & Pepper: $p \in [0.0000, 0.0250]$ (step 0.0025)
  - Bit Depth (Quantisation): $2 \to 8$ bits

## Outputs & Artefacts
- **`cache_parts/`**: Cached persistence modules and Hu moment vectors.
- **`perf_log.jsonl`**: Raw cross-validation metrics.
- **`results_*.csv`**: Tabulated accuracy drops for each stress regime.
- **`plot_results/`**:
  - `plot_panel_mechanical.png`, `plot_panel_radiometric.png`, `plot_panel_failure.png`
  - `plot_single_{protocol}.png`
  - `plot_orthogonality.png`

## Supplementary Geometric Stress Tests
Standalone scripts to isolate grid interpolation artefacts ("The Discretisation Gap"). Outputs write to `wobble/`.

**Rotational Shattering ($\textbf{H}_0$):**
Rotates a linear array of 30 isolated pixels. Demonstrates spurious component merging under bilinear interpolation.

```bash
python experiment_shattering_binary.py
```

**Loop Closure ($\textbf{H}_1$):**
Rotates five thin rectangular loops. Demonstrates feature erasure and fragmentation at non-grid-aligned angles.

```bash
python experiment_loops_binary.py
```
