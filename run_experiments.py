"""
Run every subset on every fold, one after another. Finished runs are skipped,
so rerun the same command after any interruption.

  python run_experiments.py --images C:/tmp/ds-research/images

The conjunctiva recipe is the same in both pipelines, so "specific" reuses the
"shared" C runs instead of training them twice.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from train import SUBSETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--pipelines", nargs="+", default=["shared", "specific"])
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    # Cheapest models first, so early results arrive soonest.
    subsets = sorted(args.subsets, key=lambda s: (s.count("+") + s.count("N"), SUBSETS.index(s)))
    for fold in args.folds:
        for pipeline in args.pipelines:
            for subset in subsets:
                run = Path(args.out) / pipeline / subset / f"fold{fold}"
                if (run / "summary.json").exists():
                    continue
                if pipeline == "specific" and subset == "C":
                    shared = Path(args.out) / "shared" / "C" / f"fold{fold}"
                    if (shared / "summary.json").exists():
                        shutil.copytree(shared, run, dirs_exist_ok=True)
                        continue
                print(f"=== {pipeline} {subset} fold {fold}", flush=True)
                subprocess.run([sys.executable, "train.py", "--subset", subset, "--pipeline", pipeline,
                                "--fold", str(fold), "--images", args.images, "--out", args.out,
                                "--epochs", str(args.epochs)], check=True)


if __name__ == "__main__":
    main()
