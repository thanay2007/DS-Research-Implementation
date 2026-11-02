"""
Build the one cohort table and fixed fold assignment every experiment uses.

Run from the repo root after fetch_subset.py:
  python prepare_cohort.py

Writes data/derived/cohort.csv, one row per dataset row, without any images.

What the raw table does not tell you directly:
  - patient_uuid is per capture, not per person. Field captures carry an
    _INDOOR_n / _OUTDOOR_n suffix, and 29 people appear 3 or 4 times. Folds are
    made on the id before the suffix, so nobody crosses a fold.
  - The base paper's field-representative group (516 images, 429 people,
    Appendix 6) is the rows with no survey_data_collection_place. It is held
    out as the external test and has fold = -1.
  - WHO labels come from cbc_hgb_g_dl with the cut-offs in the base paper's
    Appendix 4, not from hemoglobin_category (fixed adult cut-offs).
"""

import argparse
import re
from glob import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

IMAGE_COLS = ["image_conjunctiva", "image_tongue", "image_fingernails_open", "image_fingernails_closed"]
SURVEY_COLS = [
    "survey_age", "survey_gender", "survey_pregnancy_status", "survey_device_modal",
    "survey_data_collection_place", "survey_data_collection_location", "survey_data_collection_season",
    "survey_light_intensity", "survey_skin_tone_l", "survey_skin_tone_b",
    "survey_nail_polish_or_henna", "survey_finger_or_hand_examination", "survey_eye_examination",
    "survey_tongue_examination", "survey_clinical_examination",
    "survey_hemocue_venus_blood", "survey_hemocue_capillary_blood",
]
BANDS = ["normal", "mild", "moderate", "severe"]
PLACES = ("INDOOR", "OUTDOOR", "OTHER")


def age_years(text):
    """'5 years 3 months' -> 5.25. Anything else is missing."""
    match = re.fullmatch(r"\s*(\d+) years(?: (\d+) months)?\s*", str(text))
    if not match:
        return None
    return int(match[1]) + int(match[2] or 0) / 12


def who_cutoffs(age, sex, pregnant):
    """(mild_below, moderate_below, severe_below) in g/dL, base paper Appendix 4.

    The trimester is not recorded, so pregnancy takes the 1st/3rd trimester row:
    the 2nd trimester row is half a gram lower. The appendix stops at 65, so
    older adults keep the adult row. "Other" sex takes the men's edge, the
    higher one, which flags rather than clears.
    """
    if pregnant:
        return 11.0, 10.0, 7.0
    if age < 2:
        return 10.5, 9.5, 7.0
    if age < 5:
        return 11.0, 10.0, 7.0
    if age < 12:
        return 11.5, 11.0, 8.0
    if age < 15 or sex == "Female":
        return 12.0, 11.0, 8.0
    return 13.0, 11.0, 8.0


def who_band(hb, cutoffs):
    mild_below, moderate_below, severe_below = cutoffs
    if hb >= mild_below:
        return "normal"
    if hb >= moderate_below:
        return "mild"
    if hb >= severe_below:
        return "moderate"
    return "severe"


def read_table(root):
    frames = []
    for path in sorted(glob(f"{root}/*.parquet")):
        table = pq.read_table(path, columns=["patient_uuid", "cbc_hgb_g_dl", "hemoglobin_category", *SURVEY_COLS, *IMAGE_COLS])
        frame = table.drop_columns(IMAGE_COLS).to_pandas().astype(object)
        for col in IMAGE_COLS:
            frame[f"has_{col[6:]}"] = [cell is not None and cell["bytes"] is not None for cell in table[col].to_pylist()]
        frame["shard"] = Path(path).name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def exclusion(row):
    if pd.isna(row.hb):
        return "no_lab_hb"
    if pd.isna(row.age):
        return "no_metadata"
    # One row's survey columns are shifted along by one.
    if not pd.isna(row.survey_data_collection_place) and row.survey_data_collection_place not in PLACES:
        return "misaligned_survey"
    return ""


def build(root, folds, seed):
    df = read_table(root)
    df["participant_id"] = df.patient_uuid.str.slice(0, 36)
    df["hb"] = pd.to_numeric(df.cbc_hgb_g_dl, errors="coerce")
    df["age"] = df.survey_age.map(age_years)
    df["sex"] = df.survey_gender
    df["pregnant"] = df.survey_pregnancy_status == "PREGNANT"
    df["group"] = df.survey_data_collection_place.isna().map({True: "field", False: "dev"})
    df["exclude_reason"] = df.apply(exclusion, axis=1)

    usable = df.exclude_reason == ""
    cutoffs = df[usable].apply(lambda r: who_cutoffs(r.age, r.sex, r.pregnant), axis=1)
    df.loc[usable, "cut_mild"] = cutoffs.str[0]
    df.loc[usable, "cut_moderate"] = cutoffs.str[1]
    df.loc[usable, "cut_severe"] = cutoffs.str[2]
    df.loc[usable, "who_band"] = df[usable].apply(lambda r: who_band(r.hb, (r.cut_mild, r.cut_moderate, r.cut_severe)), axis=1)

    df["fold"] = -1
    dev = df[usable & (df.group == "dev")]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(dev, dev.who_band, groups=dev.participant_id)):
        df.loc[dev.index[test], "fold"] = fold
    return df.drop(columns=["survey_age", "survey_gender", "survey_pregnancy_status", "cbc_hgb_g_dl"])


def report(df):
    usable = df[df.exclude_reason == ""]
    print(f"{len(df)} rows, {df.participant_id.nunique()} people; excluded: "
          f"{df.exclude_reason[df.exclude_reason != ''].value_counts().to_dict()}")
    print(f"dev {(usable.group == 'dev').sum()} rows, field {(usable.group == 'field').sum()} rows "
          f"({usable.participant_id[usable.group == 'field'].nunique()} people)")
    print(pd.crosstab(usable.fold, usable.who_band)[BANDS].to_string())
    relabelled = usable[usable.who_band != usable.hemoglobin_category.str.replace("_anemia", "")]
    print(f"{len(relabelled)} rows change label against hemoglobin_category; under 15: "
          f"{(relabelled.age < 15).sum()}, pregnant: {relabelled.pregnant.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/subset/images")
    ap.add_argument("--out", default="data/derived/cohort.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = build(args.root, args.folds, args.seed)
    report(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
