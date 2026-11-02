"""
Crop and resize every photo into one fixed-size array per pipeline and view.

Run from the repo root after prepare_cohort.py:
  python preprocess_images.py --out C:/tmp/ds-research/images

Writes <out>/<pipeline>_<view>.npy (N x 256 x 256 x 3, uint8 CIELAB, rows in
cohort.csv order) and <out>/crop_status.csv. The arrays can be rebuilt at any
time and hold patient photos, so keep them out of git and out of synced folders.

Pipelines:
  shared    the base paper's conjunctiva recipe (crop under the white paper,
            256 x 256, CIELAB) applied to every site unchanged.
  specific  conjunctiva as above. Tongue: crop to the red region, fill in glare.
            Nails: colour-correct against the paper, crop to the fingertips,
            fill in glare.

The paper gives no crop geometry or thresholds, so the numbers here are ours.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SIZE = 256
VIEWS = {
    "conjunctiva": "image_conjunctiva",
    "tongue": "image_tongue",
    "nails_open": "image_fingernails_open",
    "nails_closed": "image_fingernails_closed",
}
PIPELINES = ("shared", "specific")


def center_square(image):
    h, w = image.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return image[top:top + side, left:left + side]


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count < 2:
        return np.zeros_like(mask, dtype=bool)
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def paper_mask(bgr):
    """The white reference sheet: bright, nearly colourless, one large region."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    mask = (lab[..., 0] > 150) & (np.abs(lab[..., 1] - 128) < 10) & (np.abs(lab[..., 2] - 128) < 16)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)).astype(bool)
    mask = largest_component(mask)
    return mask if mask.mean() >= 0.05 else None


def square_around(image, cx, cy, side):
    h, w = image.shape[:2]
    side = int(min(side, h, w))
    left = int(np.clip(cx - side / 2, 0, w - side))
    top = int(np.clip(cy - side / 2, 0, h - side))
    return image[top:top + side, left:left + side]


def crop_below_paper(bgr):
    """Turn the photo so the paper is on top, then take the square under its edge."""
    mask = paper_mask(bgr)
    if mask is None:
        return center_square(bgr), False
    edges = [mask[0].sum() / mask.shape[1], mask[:, -1].sum() / mask.shape[0],
             mask[-1].sum() / mask.shape[1], mask[:, 0].sum() / mask.shape[0]]
    turns = int(np.argmax(edges))
    if edges[turns] == 0:
        return center_square(bgr), False
    image, mask = np.rot90(bgr, turns), np.rot90(mask, turns)
    columns = np.where(mask.any(axis=0))[0]
    edge = int(np.median([np.where(mask[:, c])[0].max() for c in columns]))
    h, w = image.shape[:2]
    side = min(h - edge, w)
    if side < 64:
        return center_square(bgr), False
    left = (w - side) // 2
    return image[edge:edge + side, left:left + side].copy(), True


def fill_glare(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    glare = (lab[..., 0] > 235) & (np.abs(lab[..., 1] - 128) < 12) & (np.abs(lab[..., 2] - 128) < 12)
    if not glare.any():
        return bgr
    return cv2.inpaint(bgr, cv2.dilate(glare.astype(np.uint8), np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)


def crop_tongue(bgr):
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 1]
    otsu, _ = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    red = a > max(otsu, 135)
    mask = largest_component(cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool))
    if not 0.03 <= mask.mean() <= 0.6:
        return fill_glare(center_square(bgr)), False
    ys, xs = np.where(mask)
    side = 1.1 * max(np.ptp(xs), np.ptp(ys))
    return fill_glare(square_around(bgr, xs.mean(), ys.mean(), max(side, 64))), True


def crop_nails(bgr):
    """White-balance to the paper, then crop the fingertip end of the hand."""
    paper = paper_mask(bgr)
    if paper is None or paper.mean() > 0.97:
        return fill_glare(center_square(bgr)), False
    white = np.median(bgr[paper], axis=0)
    balanced = np.clip(bgr.astype(np.float32) * 235.0 / np.maximum(white, 1.0), 0, 255).astype(np.uint8)

    hand = largest_component(cv2.morphologyEx((~paper).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool))
    if hand.mean() < 0.05:
        return fill_glare(center_square(balanced)), False
    ys, xs = np.where(hand)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    center = points.mean(axis=0)
    axis = np.linalg.svd(points - center, full_matrices=False)[2][0]
    along = (points - center) @ axis
    # The wrist runs off the frame, so the end touching the border less is the fingertips.
    h, w = hand.shape
    border = (xs < 3) | (xs >= w - 3) | (ys < 3) | (ys >= h - 3)
    low, high = np.quantile(along, 0.02), np.quantile(along, 0.98)
    tips_high = (border & (along > 0)).sum() <= (border & (along < 0)).sum()
    tip = center + axis * (high if tips_high else low)
    length = high - low
    middle = tip - axis * (0.2 if tips_high else -0.2) * length
    return fill_glare(square_around(balanced, middle[0], middle[1], max(0.5 * length, 64))), True


def prepare(bgr, view, pipeline):
    if view == "conjunctiva" or pipeline == "shared":
        crop, ok = crop_below_paper(bgr)
    elif view == "tongue":
        crop, ok = crop_tongue(bgr)
    else:
        crop, ok = crop_nails(bgr)
    resized = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2LAB), ok


def process_shard(path):
    cv2.setNumThreads(0)
    table = pq.read_table(path, columns=list(VIEWS.values()))
    results = []
    for i in range(table.num_rows):
        row = {}
        for view, col in VIEWS.items():
            cell = table[col][i].as_py()
            data = cell["bytes"] if cell else None
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
            for pipeline in PIPELINES:
                row[pipeline, view] = (None, None) if bgr is None else prepare(bgr, view, pipeline)
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/derived/cohort.csv")
    ap.add_argument("--root", default="data/subset/images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort, usecols=["patient_uuid", "shard"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {(p, v): np.lib.format.open_memmap(out / f"{p}_{v}.npy", mode="w+", dtype=np.uint8, shape=(len(cohort), SIZE, SIZE, 3))
              for p in PIPELINES for v in VIEWS}
    status = {f"{p}_{v}": np.full(len(cohort), np.nan) for p in PIPELINES for v in VIEWS}

    shards = list(dict.fromkeys(cohort.shard))
    offsets = dict(zip(shards, np.cumsum([0] + [int((cohort.shard == s).sum()) for s in shards])))
    paths = [str(Path(args.root) / s) for s in shards]
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (shard, rows) in enumerate(zip(shards, pool.map(process_shard, paths, chunksize=4)), 1):
            for i, row in enumerate(rows, offsets[shard]):
                for (pipeline, view), (image, ok) in row.items():
                    if image is not None:
                        arrays[pipeline, view][i] = image
                        status[f"{pipeline}_{view}"][i] = float(ok)
            if n % 50 == 0 or n == len(shards):
                print(f"  {n}/{len(shards)} shards", flush=True)
    for array in arrays.values():
        array.flush()

    table = pd.DataFrame({"patient_uuid": cohort.patient_uuid, **status})
    table.to_csv(out / "crop_status.csv", index=False)
    print("Share of photos where the site's crop was found (the rest fall back to a centre crop):")
    print(table.drop(columns="patient_uuid").mean().round(3).to_string())


if __name__ == "__main__":
    main()
