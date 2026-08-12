#!/usr/bin/env python3
"""
match_pylint_to_classes.py — PRIME Class-Level Static-Analysis Validation

Runs Pylint on the source files containing labelled classes and matches
its class-design messages to PRIME's metric-based labels, then reports
Cohen's kappa — mirroring the function-level validation in the PeerJ CS paper.

Pylint message mapping:
    R0904  too-many-public-methods        -> God Class
    R0902  too-many-instance-attributes   -> Large Class (proxy)

NOTE on R0902: instance-attribute count is a PROXY for Large Class, not a
direct NLOC equivalent. Pylint has no native "class too long by LOC" check,
so agreement here is expected to be weaker than for God Class. Report it as
a proxy, not as a like-for-like tool equivalent. (This is the class-level
analogue of the High Fan-Out "no tool" limitation at function level.)

Because running Pylint over 2,797 repos is expensive, this script validates
on a STRATIFIED SAMPLE of classes (default 1,500) drawn from the labelled
parquet, balanced across the four cells (God+/-, Large+/-).

Usage:
    conda activate prime
    python3 match_pylint_to_classes.py \
        --split train \
        --sample-size 1500 \
        --repos-root /Add source of cloned repositories
"""

import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError:
    cohen_kappa_score = None

BASE = Path("") # Add source of cloned repositories

# Pylint symbol -> PRIME label column
PYLINT_MAP = {
    "too-many-public-methods": "god_class",        # R0904
    "too-many-instance-attributes": "large_class", # R0902 (proxy)
}


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced draw across (god_class, large_class) cells."""
    cells = []
    for g in (0, 1):
        for l in (0, 1):
            cell = df[(df["god_class"] == g) & (df["large_class"] == l)]
            cells.append(cell)
    per_cell = max(1, n // 4)
    parts = []
    for cell in cells:
        take = min(len(cell), per_cell)
        if take:
            parts.append(cell.sample(take, random_state=seed))
    out = pd.concat(parts).drop_duplicates(
        subset=["project_name", "file_path", "class_name"])
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# Pylint
# ----------------------------------------------------------------------
def run_pylint(file_path: Path) -> list:
    """Return list of pylint message dicts for a single file (JSON reporter)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pylint",
             "--output-format=json",
             "--disable=all",
             "--enable=R0904,R0902",
             str(file_path)],
            capture_output=True, text=True, timeout=120,
        )
        if not proc.stdout.strip():
            return []
        return json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def pylint_classes_in_file(messages: list) -> dict:
    """
    Map pylint messages to per-class flags.
    Pylint reports the offending class via the `obj` field.
    Returns {class_name: {"god_class": 0/1, "large_class": 0/1}}.
    """
    out = defaultdict(lambda: {"god_class": 0, "large_class": 0})
    for m in messages:
        sym = m.get("symbol")
        if sym not in PYLINT_MAP:
            continue
        cls = m.get("obj") or ""
        # `obj` may be "ClassName" or "ClassName.method"; take the class part.
        cls = cls.split(".")[0] if cls else ""
        if cls:
            out[cls][PYLINT_MAP[sym]] = 1
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--sample-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repos-root", required=True,
                    help="Root dir containing the cloned PRIME_Repos.")
    args = ap.parse_args()

    repos_root = Path(args.repos_root)
    in_path = BASE / f"{args.split}_class_labelled.parquet"
    df = pd.read_parquet(in_path)

    sample = stratified_sample(df, args.sample_size, args.seed)
    print(f"Sampled {len(sample):,} classes from {args.split} "
          f"({in_path.name})")

    # Cache pylint results per file so we don't re-run on shared files.
    file_cache = {}
    rows = []
    missing_files = 0

    for _, r in sample.iterrows():
        proj, rel, cname = r["project_name"], r["file_path"], r["class_name"]
        # file_path may be absolute or repo-relative; resolve sensibly.
        candidate = Path(rel)
        if not candidate.is_absolute():
            candidate = repos_root / proj / rel
        if not candidate.exists():
            missing_files += 1
            continue

        key = str(candidate)
        if key not in file_cache:
            file_cache[key] = pylint_classes_in_file(run_pylint(candidate))
        pl = file_cache[key].get(str(cname),
                                 {"god_class": 0, "large_class": 0})

        rows.append({
            "project_name": proj, "class_name": cname,
            "prime_god": int(r["god_class"]),
            "prime_large": int(r["large_class"]),
            "pylint_god": pl["god_class"],
            "pylint_large": pl["large_class"],
        })

    if missing_files:
        print(f"  ({missing_files} source files not found, skipped)")

    res = pd.DataFrame(rows)
    if res.empty:
        print("No classes matched to source files. Check --repos-root and "
              "file_path format.")
        return

    res.to_parquet(BASE / f"{args.split}_pylint_match.parquet", index=False)

    # ------------------------------------------------------------------
    # Agreement reporting
    # ------------------------------------------------------------------
    def report(label: str, prime_col: str, pylint_col: str, proxy=False):
        a, b = res[prime_col], res[pylint_col]
        n = len(res)
        agree = int((a == b).sum())
        print(f"\n--- {label}{' (PROXY)' if proxy else ''} ---")
        print(f"  n = {n:,}, raw agreement = {agree/n:.3f}")
        if cohen_kappa_score is not None and a.nunique() > 1 and b.nunique() > 1:
            k = cohen_kappa_score(a, b)
            cm = confusion_matrix(a, b, labels=[0, 1])
            print(f"  Cohen's kappa = {k:.3f}")
            print(f"  confusion [tn fp; fn tp] = "
                  f"[{cm[0,0]} {cm[0,1]}; {cm[1,0]} {cm[1,1]}]")
            # Precision/recall treating PRIME as reference
            tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
            prec = tp/(tp+fp) if (tp+fp) else float("nan")
            rec  = tp/(tp+fn) if (tp+fn) else float("nan")
            print(f"  (pylint vs PRIME) precision={prec:.3f} recall={rec:.3f}")
        else:
            print("  kappa undefined (a class is constant in the sample)")

    report("God Class  (R0904)", "prime_god", "pylint_god")
    report("Large Class (R0902)", "prime_large", "pylint_large", proxy=True)

    print(f"\nWrote {BASE / (args.split + '_pylint_match.parquet')}")
    print("Done.")


if __name__ == "__main__":
    main()