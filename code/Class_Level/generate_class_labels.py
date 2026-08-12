#!/usr/bin/env python3
"""
generate_class_labels.py — PRIME Class-Level Structural PDS Detection

Aggregates function-level data to class level, applies God Class and
Large Class thresholds, and writes class-level labelled parquet files.

Thresholds:
    God Class   : n_public_methods > 30
    Large Class : total_nloc > 53   (75th pct of TRAINING class distribution)

Usage:
    conda activate prime
    python3 generate_class_labels.py
"""

import argparse
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BASE = Path("") # Add source of labelled parquet files
OUT  = BASE / "class_level"

SPLITS = {
    "train": BASE / "train_labelled.parquet",
    "val":   BASE / "val_labelled.parquet",
    "test":  BASE / "test_labelled.parquet",
}

KEY_COLS = ["project_name", "file_path", "class_name"]
NEEDED = KEY_COLS + [
    "function_name", "nloc", "cyclomatic_complexity",
    "num_parameter", "num_token", "start_line", "end_line",
]

GOD_CLASS_PUBLIC_METHODS = 30      # strict >
# Large Class NLOC threshold is computed from TRAIN unless overridden.
LARGE_CLASS_NLOC_DEFAULT = 53


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def is_public(name: str) -> bool:
    return not str(name).startswith("_")


def aggregate_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse method rows into one row per class."""
    df = df[df["class_name"].notna()].copy()

    # Only keep the columns that actually exist in this parquet.
    cols = [c for c in NEEDED if c in df.columns]
    df = df[cols]

    grouped = df.groupby(KEY_COLS, sort=False)

    agg_spec = {
        "total_nloc":        ("nloc", "sum"),
        "n_methods":         ("function_name", "count"),
        "n_public":          ("function_name",
                              lambda x: sum(is_public(n) for n in x)),
        "total_cc":          ("cyclomatic_complexity", "sum"),
        "mean_cc":           ("cyclomatic_complexity", "mean"),
        "max_cc":            ("cyclomatic_complexity", "max"),
        "total_params":      ("num_parameter", "sum"),
    }
    # Drop agg entries whose source column is missing.
    agg_spec = {k: v for k, v in agg_spec.items() if v[0] in df.columns}

    class_df = grouped.agg(**agg_spec).reset_index()
    class_df["n_private"] = class_df["n_methods"] - class_df["n_public"]

    if {"start_line", "end_line"}.issubset(df.columns):
        spans = grouped.agg(class_start=("start_line", "min"),
                            class_end=("end_line", "max")).reset_index()
        class_df = class_df.merge(spans, on=KEY_COLS, how="left")

    return class_df


def apply_labels(class_df: pd.DataFrame, nloc_thr: float) -> pd.DataFrame:
    class_df["god_class"]   = (class_df["n_public"] > GOD_CLASS_PUBLIC_METHODS).astype(int)
    class_df["large_class"] = (class_df["total_nloc"] > nloc_thr).astype(int)
    class_df["anti_pattern_label"] = (
        (class_df["god_class"] == 1) | (class_df["large_class"] == 1)
    ).astype(int)
    return class_df


def summarise(name: str, class_df: pd.DataFrame, nloc_thr: float) -> None:
    n = len(class_df)
    god   = int(class_df["god_class"].sum())
    large = int(class_df["large_class"].sum())
    both  = int(((class_df["god_class"] == 1) &
                 (class_df["large_class"] == 1)).sum())
    any_  = int(class_df["anti_pattern_label"].sum())
    print(f"\n[{name}] classes={n:,}")
    print(f"  God Class   (>{GOD_CLASS_PUBLIC_METHODS} public): {god:,} ({god/n:.1%})")
    print(f"  Large Class (NLOC>{nloc_thr:g})        : {large:,} ({large/n:.1%})")
    print(f"  Both                          : {both:,} ({both/n:.1%})")
    print(f"  Any anti-pattern              : {any_:,} ({any_/n:.1%})")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nloc-threshold", type=float, default=None,
                    help="Override Large Class NLOC threshold. "
                         "Default: 75th pct of TRAIN class distribution.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Build train first to derive the threshold.
    print("Reading train ...")
    train_fn = pd.read_parquet(SPLITS["train"],
                               columns=[c for c in NEEDED
                                        if c in pd.read_parquet(
                                            SPLITS["train"]).columns])
    train_cls = aggregate_classes(train_fn)

    if args.nloc_threshold is not None:
        nloc_thr = args.nloc_threshold
    else:
        nloc_thr = float(train_cls["total_nloc"].quantile(0.75))
    print(f"Large Class NLOC threshold (75th pct train): {nloc_thr:g}")

    # 2) Label and write every split with the SAME (train-derived) threshold.
    for name, path in SPLITS.items():
        if name == "train":
            cls = train_cls
        else:
            print(f"Reading {name} ...")
            fn = pd.read_parquet(path)
            cls = aggregate_classes(fn)

        cls = apply_labels(cls, nloc_thr)
        summarise(name, cls, nloc_thr)

        out_path = OUT / f"{name}_class_labelled.parquet"
        cls.to_parquet(out_path, index=False)
        print(f"  -> wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()