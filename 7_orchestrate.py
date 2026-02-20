# 7_orchestrate.py
"""Run the pipeline."""

from __future__ import annotations

import pathlib

import fives_shared as fs


def main() -> None:
    """Run steps 1 through 6 in order.

    Returns
    -------
    None

    """
    root = pathlib.Path(__file__).resolve().parent
    steps: tuple[tuple[str, str], ...] = (
        ("Step 1 (Precompute)", "1_precompute.py"),
        ("Step 2 (Audit)", "2_audit.py"),
        ("Step 3 (Signal)", "3_signal.py"),
        ("Step 4 (Generalise)", "4_generalise.py"),
        ("Step 5 (Ablate)", "5_ablate.py"),
        ("Step 6 (Plot)", "6_plot.py"),
    )

    for label, filename in steps:
        fs.run_step(root / filename, label)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
