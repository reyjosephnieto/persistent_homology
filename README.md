# TDA Pipeline

## Requirements
- Python 3.9+
- Packages: numpy, scipy, scikit-learn, opencv-python, gudhi, matplotlib

Install dependencies:
```bash
pip install numpy scipy scikit-learn opencv-python gudhi matplotlib
```

## Dataset Setup
1. Download the FIVES dataset from:
   https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1?file=34969398
2. Extract the dataset into the workspace root.
   Keep the folder name and structure unchanged:
   `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/`
3. Keep `mask_circle.png` in the workspace root.

## Run the Pipeline
1. Place all `.py` files from this codebase in the workspace root.
2. From the workspace root, run:
```bash
python 7_orchestrate.py
```

## Outputs
- `cache_parts/`: cached persistence and geometry streams
- `perf_log.jsonl`: metrics log
- `stress_test_results.csv` and `stress_test_results_*.csv`: stress test tables
- `images/stress_test_*.png`: plots for each protocol

## Additional Experiments
Two standalone scripts show how rotation + thresholding affects simple binary shapes.

### 1) Points under rotation — `experiment_shattering_binary.py`
Rotates 30 isolated pixels (1-pixel gaps) and measures how many components remain after thresholding.

Run:
```bash
python experiment_shattering_binary.py
```

Outputs (in `wobble/`):
- `shattering_binary.pkl`: angle vs component count
- `shattering_binary.png`: plot with reference lines

### 2) Thin loops under rotation — `experiment_loops_binary.py`
Rotates five thin rectangular loops and measures how many holes remain after thresholding.

Run:
```bash
python experiment_loops_binary.py
```

Outputs (in `wobble/`):
- `loops_binary.pkl`: angle vs hole count
- `loops_binary.png`: plot with reference lines
