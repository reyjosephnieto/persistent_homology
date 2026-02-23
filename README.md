# Computational Stability of Cubical Homology

This repository contains the data processing, feature extraction, and evaluation pipeline for evaluating the stability of Topological Data Analysis (TDA) on discrete sensor grids.

Using Diabetic Retinopathy screening as the clinical testbed, the pipeline benchmarks a 6D topological summary ($\textbf{H}_0\textbf{H}_1\textbf{H}_S$) against a 7D geometric baseline (Hu Moments) across 116 distinct perturbation regimes. The pipeline tests the **Orthogonality Hypothesis**: geometric invariants provide robust Mechanical Control (affine stability), while topological invariants provide robust Radiometric Control (illumination/quantisation stability).

## Requirements
- Python 3.9+
- Packages: `numpy`, `scipy`, `scikit-learn`, `opencv-python`, `gudhi`, `matplotlib`

    pip install numpy scipy scikit-learn opencv-python gudhi matplotlib

## Dataset Setup
1. [Download the FIVES dataset (Fundus Image Vessel Segmentation)](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169?file=34969398).
2. Extract the archive directly into the workspace root.
3. The target directory must be named exactly:
   `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/`

The ingest script filters out AMD and Glaucoma partitions to isolate the balanced Normal/DR cohort ($N=400$).

## Feature Definitions
The pipeline enforces a coarse vectorisation strategy to guarantee Lipschitz continuity under the bottleneck distance:
- **Topological Vector (6D):** Concatenation of $\textbf{H}_0$ (sublevel/dark lesions), $\textbf{H}_1$ (sublevel/vascular loops), and $\textbf{H}_S$ (superlevel/bright exudates). Each homological dimension is summarised as a 2D vector: `[Top-5 Persistence Sum, Total Persistence]`. Betti counts are strictly excluded to maintain continuous stability.
- **Geometric Control (7D):** Log-transformed Hu Invariant Moments.
- **Audit Vector (13D):** $\textbf{H}_0 \oplus \textbf{H}_1 \oplus \textbf{H}_S \oplus \text{Hu}$.
Notation: $\textbf{H}_0$ denotes 0th homology on sublevel sets, $\textbf{H}_1$ denotes 1st homology on sublevel sets, and $\textbf{H}_S$ denotes 0th homology on superlevel sets (S avoids a notational clash).

## Pipeline Execution

### 1. One-Time Ingest (Staging)
Run the staging script to construct the base arrays:

    python 0_ingest.py

Outputs generated in `data/`:
- `clean_cohort.npz` (images, labels, indices)
- `clean_images.npy`, `clean_labels.npy` (CLAHE applied)
- `raw_images.npy` (raw green channel, no CLAHE, no resize)
- `raw_paths.npy` (string paths aligned with stacked order)
- `train_indices.npy`, `test_indices.npy`

### 2. Full Orchestration
Execute the entire feature extraction and stress-testing pipeline:

    python 7_orchestrate.py

The orchestrator executes the following stages sequentially:
1. **`1_precompute.py`**: Builds cached persistence/geometry features for each perturbation. Bypasses existing caches. Non-resolution protocols use `clean_images.npy`. Resolution protocols read from `raw_images.npy` and apply CLAHE post-resize.
2. **`2_audit.py`**: Conducts statistical feature separability audits (Welch's t-test, Cohen’s d) on the baseline training set, then reports mean/median/mode lifetimes for H0/H1/HS.
3. **`3_signal.py`**: Establishes baseline discriminative power via 5-fold cross-validation on the standard train cache.
4. **`4_generalise.py`**: Evaluates baseline train/test generalisation.
5. **`5_ablate.py`**: Executes the sensitivity audit. Runs Stratified 5-Fold CV on the pooled cohort ($N=400$) across all perturbations.
6. **`6_plot.py`**: Renders the robustness gap plots and differential failure matrices.

## Supplementary Data: Exploratory Audit and Preprocessing

### Mitigating Pre-existing Domain Shift
A prerequisite for measuring topological degradation is a functional, stable baseline model. A preliminary validation using a linear probe trained on the official FIVES training split ($N=300$) achieved high training accuracy ($84.0\%$) but collapsed catastrophically on the official test split ($N=100$), yielding an accuracy of $41.0\%$ (worse than random guessing).

This discrepancy indicates a severe, pre-existing domain shift between the official FIVES partitions, implying that the fixed split forces the classifier to overfit to sensor-specific hardware characteristics. To isolate the representational power of the features from these sampling artefacts, the partitions were pooled into a single balanced cohort ($N=400$, 200 Normal / 200 DR) evaluated under Stratified $5$-Fold Cross-Validation.

### Topological Hyperparameter Derivation
To systematically determine the truncation parameter $\tau$, the global distribution of feature persistences across the training cohort was evaluated. The summary statistics of the raw persistent generators are detailed below:

| **Homology** | **Mean** | **Median** | **Mode** | **Count** |
| :--- | ---: | ---: | ---: | ---: |
| $\textbf{H}_0$ | 4.080 | 3.412 | 1.482 | 8,762,399 |
| $\textbf{H}_1$ | 3.852 | 3.383 | 1.477 | 11,609,642 |
| $\textbf{H}_S$ | 4.010 | 3.422 | 1.484 | 8,668,531 |

The generation of millions of features per homology group indicates the dense presence of high-frequency stochastic noise ("topological dust"). Because the mean persistence across all groups approximates $4.0$, a strict truncation threshold of $\tau = 5.0$ was selected. This aggressively prunes transient, low-amplitude generators that are statistically probable to represent noise rather than anatomical structures.

### Statistical Audit of the Feature Space
Prior to invoking the linear probe under perturbation, a rigorous statistical audit was performed to quantify the intrinsic discriminative power of the chosen invariants on the standard training cache ($N=300$). Welch's $t$-test and Cohen's $d$ were employed to measure the separation between class means in units of pooled standard deviation.

| **Feature Name** | **Healthy** ($\mu$) | **Diabetic** ($\mu$) | **$p$-value** | **Cohen's $d$** |
| :--- | :---: | :---: | :---: | :---: |
| $\textbf{H}_S$ Top-5 Sum | 363.807 | 609.067 | $< 0.001$ | **1.573** |
| $\textbf{H}_0$ Top-5 Sum | 231.300 | 543.800 | $< 0.001$ | **1.415** |
| Hu Moment $\phi_5$ | 7.680 | -1.920 | $< 0.001$ | -0.901 |
| $\textbf{H}_1$ Top-5 Sum | 466.493 | 547.920 | $< 0.001$ | **0.879** |
| $\textbf{H}_0$ Total Persistence | 127244.039 | 111088.070 | $< 0.001$ | -0.698 |
| $\textbf{H}_S$ Total Persistence | 122674.039 | 109061.906 | $< 0.001$ | -0.641 |
| $\textbf{H}_1$ Total Persistence | 158138.562 | 139959.438 | $< 0.001$ | -0.580 |
| Hu Moment $\phi_4$ | -9.832 | -9.995 | 0.004 | -0.336 |
| Hu Moment $\phi_7$ | -0.800 | -3.840 | 0.025 | -0.260 |
| Hu Moment $\phi_3$ | -11.274 | -11.196 | 0.116 | 0.182 |
| Hu Moment $\phi_2$ | -8.490 | -8.590 | 0.165 | -0.161 |
| Hu Moment $\phi_1$ | -2.634 | -2.642 | 0.386 | -0.100 |
| Hu Moment $\phi_6$ | -2.878 | -2.078 | 0.557 | 0.068 |

**Table 1: Univariate Feature Ranking.** Features ranked by absolute effect size magnitude ($|d|$) using the $N=300$ training cohort.

The statistical results reveal that the Top-5 Persistence Sum ($S_5$) operates as the dominant signal carrier. The geometric features appear comparatively weak, with the exception of Hu Moment $\phi_5$. This confirms that Hu moments are engineered for global shape invariance, validating their selection as the global Mechanical Control vector despite weaker baseline classification utility.

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

    python experiment_shattering_binary.py

**Loop Closure ($\textbf{H}_1$):**
Rotates five thin rectangular loops. Demonstrates feature erasure and fragmentation at non-grid-aligned angles.

    python experiment_loops_binary.py
