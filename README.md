# Computational Stability of Cubical Homology

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
| Step | Script | What it does | Main inputs | Main outputs |
| :---: | :--- | :--- | :--- | :--- |
| 0 | `0_ingest.py` | Builds staged tensors from FIVES, extracts green channel, applies CLAHE (clean branch), and stores train/test index maps. | `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/` | `data/clean_cohort.npz`, `data/clean_images.npy`, `data/clean_labels.npy`, `data/raw_images.npy`, `data/raw_paths.npy`, `data/train_indices.npy`, `data/test_indices.npy` |
| 1 | `1_precompute.py` | Computes persistence diagrams (`H0/H1/HS`) and Hu vectors for each split, protocol, and perturbation level. | `data/*.npy`, protocol definitions in `fives_shared.py` | `cache_parts/{split}_{protocol}_{level}.pkl` |
| 2 | `2_audit.py` | Runs baseline univariate audit (Welch's t-test + Cohen's d) and lifetime summaries on train/standard cache. | `cache_parts/train_standard_0.pkl` | Console markdown tables, `perf_log.jsonl` records (`2_audit`, `2_audit_lifetimes`) |
| 3 | `3_signal.py` | Measures baseline predictive signal using stratified 5-fold CV for `H0`, `H1`, `HS`, `H0H1HS`, and `Hu`. | `cache_parts/train_standard_0.pkl` | `cache_parts/ablation_features.pkl`, `perf_log.jsonl` (`3_signal`) |
| 4 | `4_generalise.py` | Trains on train/standard and evaluates on test/standard (topological `H0+H1` sanity check). | `cache_parts/train_standard_0.pkl`, `cache_parts/test_standard_0.pkl` | `perf_log.jsonl` (`4_generalise`) |
| 5 | `5_ablate.py` | Main stress audit: fit on clean folds, evaluate on matched perturbed folds across all protocols/levels. | baseline + perturbed caches from step 1 | `results_mechanical.csv`, `results_radiometric.csv`, `results_failure.csv`, `stress_test_results_*.csv`, `perf_log.jsonl` (`5_ablate`) |
| 6 | `6_plot.py` | Generates grouped panels, per-protocol plots, and orthogonality plots from latest `5_ablate` run. | `perf_log.jsonl` | `plot_results/*.png` |
| 7 | `7_orchestrate.py` | Runs steps `1 -> 6` sequentially in a single command. | Python scripts and staged data | End-to-end artefacts above |

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

## Supplementary Notes (Condensed)
- **Why pooled cross-validation?** A quick train/test check on the official split showed severe drift (train accuracy $84.0\%$, test accuracy $41.0\%$), so stress evaluation is reported on pooled Normal/DR samples under stratified 5-fold CV.
- **Why truncation $\tau = 5.0$?** Lifetime distributions for `H0/H1/HS` cluster near mean persistence $\approx 4.0$, so $\tau = 5.0$ removes low-amplitude topological dust.

Lifetime summary (train baseline):

| **Homology** | **Mean** | **Median** | **Mode** | **Count** |
| :--- | ---: | ---: | ---: | ---: |
| $\textbf{H}_0$ | 4.080 | 3.412 | 1.482 | 8,762,399 |
| $\textbf{H}_1$ | 3.852 | 3.383 | 1.477 | 11,609,642 |
| $\textbf{H}_S$ | 4.010 | 3.422 | 1.484 | 8,668,531 |

Top discriminative features by absolute effect size $|d|$ (train baseline):

| **Feature** | **Cohen's $d$** |
| :--- | ---: |
| $\textbf{H}_S$ Top-5 Sum | 1.573 |
| $\textbf{H}_0$ Top-5 Sum | 1.415 |
| Hu Moment $\phi_5$ | -0.901 |
| $\textbf{H}_1$ Top-5 Sum | 0.879 |
| $\textbf{H}_0$ Total Persistence | -0.698 |

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
