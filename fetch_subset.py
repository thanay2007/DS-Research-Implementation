"""
Download a size-capped subset of sewa-rural-care/anemia-survey-dataset.

The full dataset is ~650 GB (734 parquet shards x 8 patients). Almost all of
that is `ppg_frames` (~900 JPEGs per patient, ~120 MB each). Parquet stores
each column separately, so we fetch only the columns we need via HTTP range
requests and stop before the byte budget is exceeded.

Modes (per patient, approx.):
  tabular  survey_* + cbc_* + labels + camera/PPG metadata   ~80 KB  -> all ~5.9k patients fit in 1 GB
  images   tabular + 4 clinical photos                       ~0.8 MB -> ~1,100 patients per 1 GB
  full     everything incl. PPG frames                       ~120 MB -> ~8 patients per 1 GB

Shards are picked in a seeded random order so the sample is spread across the
whole collection period rather than just the first shards.

Usage:
  python fetch_subset.py --mode images --budget-gb 1
  python fetch_subset.py --mode tabular
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/sewa-rural-care/anemia-survey-dataset"
IMAGE_COLS = ["image_conjunctiva", "image_fingernails_open", "image_fingernails_closed", "image_tongue"]
HEAVY_COLS = IMAGE_COLS + ["ppg_frames"]

fs = HfFileSystem()


class CountingFile:
    """Wraps a remote file so we know exactly how many bytes were fetched."""

    def __init__(self, f):
        self._f = f
        self.bytes_read = 0

    def read(self, n=-1):
        data = self._f.read(n)
        self.bytes_read += len(data)
        return data

    def __getattr__(self, name):
        return getattr(self._f, name)


def open_remote(path):
    # cache_type="none": fetch exactly the ranges pyarrow asks for, no read-ahead
    # (read-ahead would silently pull in neighbouring PPG bytes).
    return CountingFile(fs.open(path, "rb", cache_type="none"))


def read_footer(path):
    f = open_remote(path)
    try:
        md = pq.ParquetFile(f).metadata
        sizes = {}
        for rg in range(md.num_row_groups):
            r = md.row_group(rg)
            for c in range(r.num_columns):
                col = r.column(c)
                top = col.path_in_schema.split(".")[0]
                sizes[top] = sizes.get(top, 0) + col.total_compressed_size
        return {"rows": md.num_rows, "col_bytes": sizes, "footer_bytes": f.bytes_read}
    finally:
        f.close()


def columns_for(mode, all_cols):
    if mode == "full":
        return list(all_cols)
    if mode == "images":
        return [c for c in all_cols if c != "ppg_frames"]
    return [c for c in all_cols if c not in HEAVY_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tabular", "images", "full"], default="images")
    ap.add_argument("--budget-gb", type=float, default=1.0)
    ap.add_argument("--out", default="data/subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out) / args.mode
    out.mkdir(parents=True, exist_ok=True)
    budget = int(args.budget_gb * 1e9)

    shards = sorted(p for p in fs.ls(f"{REPO}/data", detail=False) if p.endswith(".parquet"))
    random.Random(args.seed).shuffle(shards)

    # Footers are ~65 KB each, so only read as many as the budget needs, and
    # cache them so re-runs don't refetch.
    footer_cache = Path(args.out) / "footers.json"
    footers = json.loads(footer_cache.read_text()) if footer_cache.exists() else {}

    # Plan: take shards in random order until the next one would overflow the budget.
    # Footer bytes are counted too, so the cap covers everything downloaded.
    plan, planned, full = [], 0, False
    batch = args.workers * 4
    for start in range(0, len(shards), batch):
        chunk = shards[start:start + batch]
        missing = [s for s in chunk if s not in footers]
        with ThreadPoolExecutor(args.workers) as ex:
            for s, meta in zip(missing, ex.map(read_footer, missing)):
                footers[s] = meta
        for s in chunk:
            meta = footers[s]
            cols = columns_for(args.mode, meta["col_bytes"])
            # footer is read twice: once here for planning, once again in fetch()
            cost = 2 * meta["footer_bytes"] + sum(meta["col_bytes"][c] for c in cols)
            if planned + cost > budget:
                full = True
                break
            plan.append((s, cols))
            planned += cost
        if full:
            break
    footer_cache.write_text(json.dumps(footers))

    n_rows = sum(footers[s]["rows"] for s, _ in plan)
    print(f"Mode={args.mode}: {len(plan)}/{len(shards)} shards, {n_rows} patients, "
          f"~{planned / 1e9:.3f} GB planned (budget {args.budget_gb} GB)")

    def fetch(item):
        s, cols = item
        dest = out / Path(s).name
        if dest.exists():
            return 0
        f = open_remote(s)
        try:
            table = pq.ParquetFile(f).read(columns=cols)
        finally:
            f.close()
        tmp = dest.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(dest)
        return f.bytes_read

    fetched = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, b in enumerate(ex.map(fetch, plan), 1):
            fetched += b
            if i % 25 == 0 or i == len(plan):
                print(f"  {i}/{len(plan)} shards, {fetched / 1e9:.3f} GB downloaded this run")

    print(f"Done. Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
