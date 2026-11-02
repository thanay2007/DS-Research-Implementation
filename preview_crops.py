"""
Save one contact sheet per pipeline and view, to check the crops by eye.

  python preview_crops.py --images C:/tmp/ds-research/images

The sheets show patient photos, so they go under data/ and must not be shared
or committed (see the data use policy in the README).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from preprocess_images import PIPELINES, VIEWS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/derived/crop_previews")
    ap.add_argument("--side", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pipeline in PIPELINES:
        for view in VIEWS:
            array = np.load(Path(args.images) / f"{pipeline}_{view}.npy", mmap_mode="r")
            rows = np.sort(np.random.default_rng(args.seed).choice(len(array), args.side ** 2, replace=False))
            cells = [cv2.resize(cv2.cvtColor(np.array(array[i]), cv2.COLOR_LAB2BGR), (128, 128)) for i in rows]
            sheet = np.vstack([np.hstack(cells[i:i + args.side]) for i in range(0, len(cells), args.side)])
            cv2.imwrite(str(out / f"{pipeline}_{view}.png"), sheet)
    print(f"Saved contact sheets to {out.resolve()}")


if __name__ == "__main__":
    main()
