"""
Load the local subset written by fetch_subset.py as a Hugging Face Dataset.

    from load_subset import load_subset
    ds = load_subset("images")
    ds[0]["image_tongue"].show()          # PIL image
"""

from collections import Counter, defaultdict
from glob import glob

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image, Sequence, load_dataset

IMAGE_COLS = ["image_conjunctiva", "image_fingernails_open", "image_fingernails_closed", "image_tongue"]


def shared_features(files):
    # A few shards store numeric columns as numbers (or all-empty ones as null)
    # where the rest use strings, so take each column's most common type.
    types = defaultdict(Counter)
    for f in files:
        for field in pq.read_schema(f):
            types[field.name][field.type] += 1
    return Features.from_arrow_schema(
        pa.schema([(name, c.most_common(1)[0][0]) for name, c in types.items()])
    )


def load_subset(mode="images", root="data/subset"):
    files = sorted(glob(f"{root}/{mode}/*.parquet"))
    ds = load_dataset("parquet", data_files=files, features=shared_features(files), split="train")
    # The shards store images as raw {bytes, path} structs; decode them as PIL images.
    for col in IMAGE_COLS:
        if col in ds.column_names:
            ds = ds.cast_column(col, Image())
    if "ppg_frames" in ds.column_names:
        ds = ds.cast_column("ppg_frames", Sequence(Image()))
    return ds


if __name__ == "__main__":
    import sys

    ds = load_subset(sys.argv[1] if len(sys.argv) > 1 else "images")
    print(ds)
    print(ds.select_columns(["hemoglobin_category"]).to_pandas()["hemoglobin_category"].value_counts(dropna=False))
