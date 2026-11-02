"""Reproduce all three projects end to end.

    python run_all.py              # everything
    python run_all.py --project 2  # just the predictive maintenance pipeline
    python run_all.py --skip-training   # regenerate data and reports using saved models

Each stage runs as a subprocess from its own project directory, exactly as documented in
that project's README, so this script is a convenience wrapper and never a second,
divergent code path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).absolute().parent

Stage = Tuple[str, List[str], bool]   # (label, module args, is_training)

PIPELINES: dict[int, dict] = {
    1: {
        "name": "Customer Churn Prediction",
        "directory": ROOT / "project1_customer_churn",
        "stages": [
            ("generate synthetic customer base", ["-m", "src.generate_data"], False),
            ("clean and validate", ["-m", "src.preprocess"], False),
            ("engineer behavioural features", ["-m", "src.features"], False),
            ("train and tune models", ["-m", "src.train"], True),
            ("evaluate and pick threshold", ["-m", "src.evaluate"], False),
            ("score base and rank risk", ["-m", "src.score_customers"], False),
        ],
    },
    2: {
        "name": "Predictive Maintenance",
        "directory": ROOT / "project2_predictive_maintenance",
        "stages": [
            ("simulate fleet telemetry", ["-m", "src.generate_data"], False),
            ("label failure windows", ["-m", "src.labeling"], False),
            ("build causal features (+ leakage check)", ["-m", "src.features"], False),
            ("train and tune models", ["-m", "src.train"], True),
            ("evaluate, lead time, cost", ["-m", "src.evaluate"], False),
            ("build fleet monitoring view", ["-m", "src.monitor"], False),
        ],
    },
    3: {
        "name": "Product Recommendation",
        "directory": ROOT / "project3_recommendation",
        "stages": [
            ("generate interaction log", ["-m", "src.generate_data"], False),
            ("inspect split and sparsity", ["-m", "src.data_prep"], False),
            ("fit models and evaluate ranking", ["-m", "src.evaluate"], True),
        ],
    },
}


def run_stage(directory: Path, arguments: List[str], label: str) -> float:
    """Run one stage, streaming its output. Raises on failure."""
    print(f"\n{'-' * 78}\n>>> {label}\n{'-' * 78}", flush=True)
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-u", *arguments], cwd=directory, check=False
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(f"stage failed ({label}) with exit code {result.returncode}")
    print(f"<<< done in {elapsed:.1f}s", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ML project pipelines.")
    parser.add_argument(
        "--project", type=int, choices=sorted(PIPELINES), action="append",
        help="run only this project (repeatable); default is all three",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="skip the model-fitting stages and reuse saved models",
    )
    arguments = parser.parse_args()
    selected = sorted(set(arguments.project)) if arguments.project else sorted(PIPELINES)

    timings: list[tuple[str, float]] = []
    overall_start = time.time()

    for number in selected:
        pipeline = PIPELINES[number]
        print(f"\n{'=' * 78}")
        print(f"PROJECT {number}: {pipeline['name']}")
        print(f"{'=' * 78}")

        for label, module_args, is_training in pipeline["stages"]:
            if is_training and arguments.skip_training:
                print(f"\n>>> skipping (--skip-training): {label}")
                continue
            elapsed = run_stage(pipeline["directory"], module_args, f"P{number}: {label}")
            timings.append((f"P{number} {label}", elapsed))

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for label, elapsed in timings:
        print(f"  {label:<52}{elapsed:>8.1f}s")
    print(f"  {'TOTAL':<52}{time.time() - overall_start:>8.1f}s")

    print("\nReports and figures:")
    for number in selected:
        print(f"  {PIPELINES[number]['directory'].name}/reports/")
    print("\nInteractive apps:")
    for number in selected:
        print(f"  cd {PIPELINES[number]['directory'].name} && streamlit run app.py")
    if 3 in selected:
        print("\nRecommendation API:")
        print("  cd project3_recommendation && uvicorn src.api:app --port 8000")


if __name__ == "__main__":
    main()
