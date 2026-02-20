# Computational Stability of Cubical Homology

This repository contains the data processing, feature extraction, and evaluation pipeline for evaluating the stability of Topological Data Analysis (TDA) on discrete sensor grids. 

Using Diabetic Retinopathy screening as the clinical testbed, the pipeline benchmarks a 6D topological summary ($\mathbf{H}_0\mathbf{H}_}1\mathbf{H}_S$) against a 7D geometric baseline (Hu Moments) across 116 distinct perturbation regimes. The framework empirically validates the **Orthogonality Hypothesis**: geometric invariants provide robust Mechanical Control (affine stability), while topological invariants provide robust Radiometric Control (illumination/quantisation stability).

## Requirements
- Python 3.9+
- Packages: `numpy`, `scipy`, `scikit-learn`, `opencv-python`, `gudhi`, `matplotlib`

    pip install numpy scipy scikit-learn opencv-python gudhi matplotlib

## Dataset Setup
1. [Download the FIVES dataset (Fundus Image Vessel Segmentation)](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169?file=34969398).
   
   
3. Extract the archive directly into the workspace root. 
4. The target directory must be named exactly:
   `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/`

*Note: The ingest script automatically filters out AMD and Glaucoma partitions to isolate the balanced Normal/DR cohort ($N=400$).*

## Feature Definitions
The pipeline enforces a coarse vectorisation strategy to guarantee Lipschitz continuity under the bottleneck distance:
- **Topological Vector (6D):** Concatenation of $\mathbf{H}_0$ (sublevel/dark lesions), $\mathbf{H}_1$ (sublevel/vascular loops), and $\mathbf{H}_S$ (superlevel/bright exudates). Each homological dimension is summarised as a 2D vector: `[Top-5 Persistence Sum, Total Persistence]`. Betti counts are strictly excluded to maintain continuous stability.
- **Geometric Control (7D):** Log-transformed Hu Invariant Moments.
- **Audit Vector (13D):** $\mathbf{H}_0 \oplus \mathbf{H}_1 \oplus \mathbf{H}_S \oplus \text{Hu}$.

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
2. **`2_audit.py`**: Conducts statistical feature separability audits (Welch's t-test, Cohen’s d) on the baseline training set.
3. **`3_signal.py`**: Establishes baseline discriminative power via 5-fold cross-validation on the standard train cache.
4. **`4_generalise.py`**: Evaluates baseline train/test generalisation.
5. **`5_ablate.py`**: Executes the sensitivity audit. Runs Stratified 5-Fold CV on the pooled cohort ($N=400$) across all perturbations.
6. **`6_plot.py`**: Renders the robustness gap plots and differential failure matrices.

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

**Rotational Shattering ($\mathbf{H}_0$):**
Rotates a linear array of 30 isolated pixels. Demonstrates spurious component merging under bilinear interpolation.

    python experiment_shattering_binary.py


**Loop Closure ($\mathbf{H}_1$):**
Rotates five thin rectangular loops. Demonstrates feature erasure and fragmentation at non-grid-aligned angles.

    python experiment_loops_binary.py
