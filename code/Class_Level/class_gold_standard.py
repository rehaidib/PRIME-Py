#!/usr/bin/env python3
"""
class_gold_standard.py — PRIME Class-Level Gold-Standard Sampler

Draws a stratified sample of classes for independent human annotation of
God Class and Large Class, and emits a blinded annotation sheet (CSV) plus
a hidden answer key (parquet) for later kappa computation.

Stratification rationale
-------------------------
God Class is validated by Pylint (kappa 0.932), so it needs only light
human confirmation. Large Class has NO tool equivalent (R0902 is a weak
proxy, kappa 0.151), so human annotation is its PRIMARY validation route
and must be well represented across the decision boundary.

We therefore stratify across BOTH labels AND a band around the Large Class
NLOC threshold, so annotators see borderline cases (where construct validity
is actually tested), not just obvious extremes.

Cells (target 96 classes, balanced):
    1. God+ Large+        (the dense overlap cell)
    2. God- Large+ far     (NLOC well above threshold)
    3. God- Large+ near    (NLOC just above threshold  -> borderline)
    4. God- Large- near    (NLOC just below threshold  -> borderline)
    5. God- Large- far     (clearly small)
    6. God+ Large-         (rare; include all available)

Usage:
    conda activate prime
    python3 class_gold_standard.py --n 96 --split train
"""

import argparse
from pathlib import Path
import pandas as pd

BASE = Path("") # Add source of labelled parquet files for classes
OUT = BASE / "gold_standard"

LARGE_THR = 54           # the locked Large Class NLOC threshold
NEAR_BAND = 0.20         # +/- 20% of threshold counts as "borderline"


def assign_cell(row) -> str:
    g, l = row["god_class"], row["large_class"]
    nloc = row["total_nloc"]
    lo, hi = LARGE_THR * (1 - NEAR_BAND), LARGE_THR * (1 + NEAR_BAND)
    near = lo <= nloc <= hi
    if g == 1 and l == 1:
        return "1_god_large"
    if g == 1 and l == 0:
        return "6_god_only"
    if g == 0 and l == 1:
        return "3_large_near" if near else "2_large_far"
    return "4_small_near" if near else "5_small_far"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(BASE / f"{args.split}_class_labelled.parquet")
    df["cell"] = df.apply(assign_cell, axis=1)

    # Target balanced allocation; rare cells contribute all they have,
    # and the remainder is redistributed across the populous cells.
    cells = sorted(df["cell"].unique())
    target = {c: args.n // len(cells) for c in cells}

    parts, shortfall = [], 0
    for c in cells:
        pool = df[df["cell"] == c]
        take = min(len(pool), target[c])
        shortfall += target[c] - take
        if take:
            parts.append(pool.sample(take, random_state=args.seed))

    sample = pd.concat(parts)
    # Redistribute shortfall into the overlap + borderline cells that matter.
    if shortfall > 0:
        priority = df[df["cell"].isin(
            ["1_god_large", "3_large_near", "4_small_near"])]
        extra = priority.drop(sample.index, errors="ignore").sample(
            min(shortfall, len(priority)), random_state=args.seed + 1)
        sample = pd.concat([sample, extra])

    sample = sample.drop_duplicates(
        subset=["project_name", "file_path", "class_name"]).reset_index(drop=True)

    # Shuffle so annotators can't infer cells from ordering.
    sample = sample.sample(frac=1, random_state=args.seed + 2).reset_index(drop=True)
    sample.insert(0, "annotation_id", range(1, len(sample) + 1))

    print(f"Gold-standard sample: {len(sample)} classes")
    print(sample["cell"].value_counts().sort_index().to_string())

    # Blinded sheet for annotators — NO labels, NO cell.
    blind_cols = ["annotation_id", "project_name", "file_path",
                  "class_name", "total_nloc", "n_methods", "n_public"]
    blind = sample[[c for c in blind_cols if c in sample.columns]].copy()
    blind["is_god_class"] = ""    # annotator fills 0/1
    blind["is_large_class"] = ""  # annotator fills 0/1
    blind["notes"] = ""
    for who in ("annotatorA", "annotatorB"):
        p = OUT / f"{args.split}_annotation_{who}.csv"
        blind.to_csv(p, index=False)
        print(f"  -> {p}")

    # Hidden key with PRIME labels + cells for later agreement analysis.
    key = sample[["annotation_id", "project_name", "file_path", "class_name",
                  "god_class", "large_class", "total_nloc", "n_public",
                  "cell"]]
    key_path = OUT / f"{args.split}_answer_key.parquet"
    key.to_parquet(key_path, index=False)
    print(f"  -> {key_path} (hidden key)")
    print("\nDone.")


if __name__ == "__main__":
    main()