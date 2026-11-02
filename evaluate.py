"""
Turn finished runs into the README's tables. Run from the repo root:
  python evaluate.py

Reads data/runs/<pipeline>/<subset>/fold*/predictions.csv and writes CSVs to
data/results/. Test-fold predictions are pooled across folds. The field group
is predicted by all five fold models, and those predictions are averaged.

Every model is compared on the rows all models could use. Patients have
different cut-offs, so the classification score is (cut-off - predicted Hb).
Bootstraps resample people, not rows, because some people have several rows.
"""

import argparse
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

SITES = ("C", "T", "N")
# The band edge a patient must fall below to count as a positive.
TASKS = {"any_anemia": "cut_mild", "moderate_or_severe": "cut_moderate", "severe": "cut_severe"}
SPECIFICITY = 0.80
# Fixed in advance: does each other site add to the conjunctiva, and does the
# full model beat the conjunctiva?
MAIN_COMPARISONS = [("C+T", "C"), ("C+N", "C"), ("C+T+N", "C")]


def load_runs(root, cohort):
    runs = {}
    for path in sorted(Path(root).glob("*/*/fold*/predictions.csv")):
        pipeline, subset = path.parts[-4], path.parts[-3]
        frame = pd.read_csv(path)
        frame = frame[frame.split.isin(["test", "field"])]
        runs.setdefault((pipeline, subset), []).append(frame)
    out = {}
    for key, frames in runs.items():
        all_rows = pd.concat(frames)
        test = all_rows[all_rows.split == "test"]
        field = all_rows[all_rows.split == "field"].groupby("patient_uuid", as_index=False).agg(
            pred=("pred", "mean"), split=("split", "first"))
        merged = pd.concat([test[["patient_uuid", "pred", "split"]], field])
        out[key] = merged.merge(cohort, on="patient_uuid", how="left")
    return out


def sensitivity_at(y, score, specificity):
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(1 - specificity, fpr, tpr))


def metrics(df):
    error = df.pred - df.hb
    row = {
        "n": len(df),
        "mae": error.abs().mean(),
        "within_1": (error.abs() <= 1).mean(),
        "bias": error.mean(),
        "loa_low": error.mean() - 1.96 * error.std(),
        "loa_high": error.mean() + 1.96 * error.std(),
    }
    for task, cut in TASKS.items():
        y = (df.hb < df[cut]).to_numpy(int)
        if 0 < y.sum() < len(y):
            score = df[cut] - df.pred
            row[f"auc_{task}"] = roc_auc_score(y, score)
            row[f"sens_at_spec{int(SPECIFICITY * 100)}_{task}"] = sensitivity_at(y, score, SPECIFICITY)
    return row


def bootstrap_people(df, statistic, n, seed):
    """Percentile CI of statistic(df), resampling whole people."""
    rng = np.random.default_rng(seed)
    people = list(df.groupby("participant_id").indices.values())
    values = []
    for _ in range(n):
        picks = rng.integers(0, len(people), len(people))
        values.append(statistic(df.iloc[np.concatenate([people[i] for i in picks])]))
    return pd.DataFrame(values).quantile([0.025, 0.975])


def delong(y, score_a, score_b):
    """Paired DeLong test: returns (AUC a - AUC b, two-sided p)."""
    y = np.asarray(y, bool)
    def components(score):
        pos, neg = score[y], score[~y]
        # Placement values, using ranks so ties count one half.
        order = np.concatenate([pos, neg])
        ranks = pd.Series(order).rank().to_numpy()
        pos_ranks = pd.Series(pos).rank().to_numpy()
        neg_ranks = pd.Series(neg).rank().to_numpy()
        v10 = (ranks[:len(pos)] - pos_ranks) / len(neg)
        v01 = 1 - (ranks[len(pos):] - neg_ranks) / len(pos)
        return v10, v01
    a10, a01 = components(np.asarray(score_a, float))
    b10, b01 = components(np.asarray(score_b, float))
    s10, s01 = np.cov(a10, b10), np.cov(a01, b01)
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(a10) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(a01)
    diff = a10.mean() - b10.mean()
    return diff, float(2 * norm.sf(abs(diff) / np.sqrt(var))) if var > 0 else 1.0


def holm(pvalues):
    p = np.asarray(pvalues, float)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def overall_table(frames, args):
    rows = []
    for (pipeline, subset, split), df in frames.items():
        row = {"pipeline": pipeline, "subset": subset, "split": split, **metrics(df)}
        ci = bootstrap_people(df, lambda d: metrics(d), args.bootstrap, args.seed)
        for col in ci.columns:
            if col != "n":
                row[f"{col}_low"], row[f"{col}_high"] = ci.loc[0.025, col], ci.loc[0.975, col]
        rows.append(row)
    return pd.DataFrame(rows)


def compare(a, b, args):
    """Paired difference a - b on the same rows. Negative MAE difference favours a."""
    joined = a.merge(b[["patient_uuid", "pred"]], on="patient_uuid", suffixes=("", "_b"))
    def statistic(d):
        out = {"mae_diff": (d.pred - d.hb).abs().mean() - (d.pred_b - d.hb).abs().mean()}
        for task, cut in TASKS.items():
            y = d.hb < d[cut]
            out[f"auc_diff_{task}"] = roc_auc_score(y, d[cut] - d.pred) - roc_auc_score(y, d[cut] - d.pred_b)
        return out
    point = statistic(joined)
    ci = bootstrap_people(joined, statistic, args.bootstrap, args.seed)
    row = {"n": len(joined), **point}
    for col in ci.columns:
        row[f"{col}_low"], row[f"{col}_high"] = ci.loc[0.025, col], ci.loc[0.975, col]
    diffs = (joined.pred - joined.hb).abs() - (joined.pred_b - joined.hb).abs()
    means = diffs.groupby(joined.participant_id).sum()
    row["mae_p"] = float(2 * norm.sf(abs(means.mean()) / (means.std() / np.sqrt(len(means)))))
    for task, cut in TASKS.items():
        y = joined.hb < joined[cut]
        row[f"delong_p_{task}"] = delong(y, joined[cut] - joined.pred, joined[cut] - joined.pred_b)[1]
    return row


def comparisons_table(frames, args):
    rows = []
    pipelines = {key[0] for key in frames}
    subsets = {key[1] for key in frames}
    for pipeline in sorted(pipelines):
        for split in ("test", "field"):
            for a, b in combinations(sorted(subsets, key=len), 2):
                if {(pipeline, a, split), (pipeline, b, split)} <= frames.keys():
                    big, small = (b, a) if len(a) <= len(b) else (a, b)
                    rows.append({"pipeline": pipeline, "split": split, "model": big, "against": small,
                                 "main": (big, small) in MAIN_COMPARISONS,
                                 **compare(frames[pipeline, big, split], frames[pipeline, small, split], args)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["mae_p_holm"] = np.nan
    for _, group in table.groupby(["pipeline", "split", "main"]):
        table.loc[group.index, "mae_p_holm"] = holm(group.mae_p)
    return table


def error_overlap_table(frames):
    """How often two single-site models fail on the same patient, against chance."""
    rows = []
    for (pipeline, subset, split), df in frames.items():
        if subset not in SITES:
            continue
        for other in SITES:
            key = (pipeline, other, split)
            if subset < other and key in frames:
                joined = df.merge(frames[key][["patient_uuid", "pred"]], on="patient_uuid", suffixes=("", "_b"))
                for task, cut in TASKS.items():
                    truth = joined.hb < joined[cut]
                    wrong_a = (joined.pred < joined[cut]) != truth
                    wrong_b = (joined.pred_b < joined[cut]) != truth
                    both, either = (wrong_a & wrong_b).mean(), (wrong_a | wrong_b).mean()
                    rows.append({"pipeline": pipeline, "split": split, "task": task, "site_a": subset, "site_b": other,
                                 "wrong_a": wrong_a.mean(), "wrong_b": wrong_b.mean(), "both_wrong": both,
                                 "expected_if_independent": wrong_a.mean() * wrong_b.mean(),
                                 "jaccard": both / either if either else np.nan})
    return pd.DataFrame(rows)


def shapley_table(overall):
    """Each site's Shapley contribution to lowering MAE and raising any-anemia AUC.

    The empty set takes the model's value for no images: the MAE of predicting
    the mean Hb on that split, and an AUC of 0.5.
    """
    rows = []
    for (pipeline, split), group in overall.groupby(["pipeline", "split"]):
        table = group.set_index("subset")
        if not all("+".join(s) in table.index for k in range(1, 4) for s in combinations(SITES, k)):
            continue
        for measure, empty, sign in (("mae", table.mae_null.iloc[0], -1.0), ("auc_any_anemia", 0.5, 1.0)):
            def value(sites):
                return empty if not sites else table.loc["+".join(s for s in SITES if s in sites), measure]
            for site in SITES:
                others = [s for s in SITES if s != site]
                total = 0.0
                for k in range(len(others) + 1):
                    for coalition in combinations(others, k):
                        weight = factorial(k) * factorial(len(SITES) - k - 1) / factorial(len(SITES))
                        total += weight * (value(coalition + (site,)) - value(coalition))
                rows.append({"pipeline": pipeline, "split": split, "measure": measure, "site": site, "shapley": sign * total})
    return pd.DataFrame(rows)


def subgroup_masks(df):
    light = pd.to_numeric(df.survey_light_intensity, errors="coerce")
    return {
        "age_under_15": df.age < 15,
        "age_15_plus": df.age >= 15,
        "pregnant": df.pregnant,
        "not_pregnant_women": (df.sex == "Female") & ~df.pregnant & (df.age >= 15),
        "men": (df.sex == "Male") & (df.age >= 15),
        **{f"band_{band}": df.who_band == band for band in ("normal", "mild", "moderate", "severe")},
        **{f"device_{device}": df.survey_device_modal == device for device in df.survey_device_modal.dropna().unique()},
        "light_low": light < light.quantile(1 / 3),
        "light_high": light >= light.quantile(2 / 3),
        # The "bad images" explanation: photos clinicians flagged for henna or polish.
        "nails_clean": (df.survey_nail_polish_or_henna == "No") & (df.survey_finger_or_hand_examination == "NONE"),
        "nails_marked": (df.survey_nail_polish_or_henna == "Yes") | (df.survey_finger_or_hand_examination == "HENNA"),
    }


def subgroup_table(frames):
    rows = []
    for (pipeline, subset, split), df in frames.items():
        for name, mask in subgroup_masks(df).items():
            part = df[mask.fillna(False)]
            if len(part) >= 30:
                rows.append({"pipeline": pipeline, "subset": subset, "split": split, "subgroup": name, **metrics(part)})
    return pd.DataFrame(rows)


def reference_table(frames):
    """HemoCue and clinician examination on the same rows, as in the base paper."""
    rows = []
    for split in ("test", "field"):
        base = next((df for key, df in frames.items() if key[2] == split), None)
        if base is None:
            continue
        for name, col in (("hemocue_venous", "survey_hemocue_venus_blood"), ("hemocue_capillary", "survey_hemocue_capillary_blood")):
            d = base.assign(pred=pd.to_numeric(base[col], errors="coerce")).dropna(subset=["pred"])
            if len(d) >= 30:
                rows.append({"reference": name, "split": split, **metrics(d)})
        call = base.survey_clinical_examination.str.upper()
        known = base[call.isin(["NO_ANEMIA", "MILD_ANEMIA", "MODERATE_ANEMIA", "SEVERE_ANEMIA"])]
        positive = known.survey_clinical_examination.str.upper() != "NO_ANEMIA"
        truth = known.hb < known.cut_mild
        rows.append({"reference": "clinician_exam", "split": split, "n": len(known),
                     "sensitivity_any_anemia": (positive & truth).sum() / truth.sum(),
                     "specificity_any_anemia": (~positive & ~truth).sum() / (~truth).sum()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--cohort", default="data/derived/cohort.csv")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort, low_memory=False)
    runs = load_runs(args.runs, cohort)
    frames = {}
    for (pipeline, subset), df in runs.items():
        for split, part in df.groupby("split"):
            frames[pipeline, subset, split] = part.reset_index(drop=True)

    # Compare every model on the rows every model has, split by split.
    for split in ("test", "field"):
        keys = [key for key in frames if key[2] == split]
        if keys:
            shared_rows = set.intersection(*(set(frames[key].patient_uuid) for key in keys))
            for key in keys:
                frames[key] = frames[key][frames[key].patient_uuid.isin(shared_rows)].reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overall = overall_table(frames, args)
    overall["mae_null"] = [(frames[p, s, sp].hb - frames[p, s, sp].hb.mean()).abs().mean()
                           for p, s, sp in zip(overall.pipeline, overall.subset, overall.split)]
    tables = {
        "overall": overall,
        "comparisons": comparisons_table(frames, args),
        "error_overlap": error_overlap_table(frames),
        "shapley": shapley_table(overall),
        "subgroups": subgroup_table(frames),
        "references": reference_table(frames),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    columns = ["pipeline", "subset", "split", "n", "mae", "mae_low", "mae_high", "within_1", "auc_any_anemia",
               "auc_moderate_or_severe", "auc_severe", "sens_at_spec80_any_anemia"]
    print(overall[columns].round(3).sort_values(["split", "pipeline", "mae"]).to_string(index=False))
    print(f"Saved {', '.join(tables)} to {out.resolve()}")


if __name__ == "__main__":
    main()
