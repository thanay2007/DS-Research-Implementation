"""
Load the local subset written by fetch_subset.py as a Hugging Face Dataset.

    from load_subset import load_subset
    ds = load_subset("images")
    ds[0]["image_tongue"].show()          # PIL image
"""

from datasets import Image, Sequence, load_dataset

IMAGE_COLS = ["image_conjunctiva", "image_fingernails_open", "image_fingernails_closed", "image_tongue"]


def load_subset(mode="images", root="data/subset"):
    ds = load_dataset("parquet", data_files=f"{root}/{mode}/*.parquet", split="train")
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
