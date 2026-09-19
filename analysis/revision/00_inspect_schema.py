"""Step 0: inspect the labelled Parquet splits and validate config.py.

Prints, for each split file: existence, row count (from Parquet metadata,
no data loaded), and the full column list.  Then checks every mapping in
config.COLUMNS / config.LABELS and suggests candidates for anything that
does not match, using simple keyword heuristics.

Usage:
    conda activate prime
    python 00_inspect_schema.py

Paste the full output back into the chat so the mapping can be finalized.
"""
from __future__ import annotations
import sys
import pyarrow.parquet as pq

import config

KEYWORDS = {
    "project":  ["project", "repo"],
    "nloc":     ["nloc"],
    "cc":       ["cyclomatic", "ccn", "complexity"],
    "params":   ["param", "arg"],
    "branches": ["branch"],
    "calls":    ["fan_out", "fanout", "outgoing", "call"],
    "body":     ["body", "source", "code", "snippet", "text"],
    # label columns
    "long_method":         ["long_method"],
    "high_cc":             ["high_cc", "high_complexity", "complexity"],
    "long_parameter_list": ["parameter_list", "long_param"],
    "excessive_branching": ["spaghetti", "branch"],
    "high_fan_out":        ["fan_out", "fanout"],
}


def suggest(logical: str, columns: list[str]) -> list[str]:
    kws = KEYWORDS.get(logical, [logical])
    hits = [c for c in columns if any(k in c.lower() for k in kws)]
    return hits[:6]


def main() -> int:
    print(f"DATA_DIR = {config.DATA_DIR}")
    if not config.DATA_DIR.exists():
        print("!! DATA_DIR does not exist -- fix the path in config.py")
        print("   Directory contents cannot be listed; aborting.")
        return 1

    print("\nFiles present in DATA_DIR:")
    for p in sorted(config.DATA_DIR.iterdir()):
        print(f"  {p.name}")

    all_columns: list[str] = []
    total_rows = 0
    print("\n--- split files ---")
    for split, fname in config.SPLIT_FILES.items():
        path = config.DATA_DIR / fname
        if not path.exists():
            print(f"[{split:10s}] MISSING: {fname}")
            continue
        meta = pq.read_metadata(path)
        schema = pq.read_schema(path)
        cols = schema.names
        all_columns = cols  # assume identical schemas across splits
        total_rows += meta.num_rows
        print(f"[{split:10s}] {fname}: {meta.num_rows:,} rows, "
              f"{len(cols)} columns")

    if not all_columns:
        print("!! No split file could be read -- fix SPLIT_FILES in config.py")
        return 1

    print(f"\nTotal rows across splits: {total_rows:,} "
          f"(expected 1,997,535 -> {'OK' if total_rows == 1_997_535 else 'MISMATCH'})")

    print("\nAll columns (name : dtype):")
    schema = pq.read_schema(config.DATA_DIR / next(iter(config.SPLIT_FILES.values())))
    for field in schema:
        print(f"  {field.name} : {field.type}")

    print("\n--- config.COLUMNS validation ---")
    problems = 0
    for logical, actual in config.COLUMNS.items():
        if actual in all_columns:
            print(f"  {logical:10s} -> '{actual}'  OK")
        else:
            problems += 1
            cands = suggest(logical, all_columns)
            print(f"  {logical:10s} -> '{actual}'  NOT FOUND"
                  f"   candidates: {cands or '(none found)'}")

    print("\n--- config.LABELS validation (optional columns) ---")
    for logical, actual in config.LABELS.items():
        if actual is None:
            print(f"  {logical:22s} -> (disabled)")
        elif actual in all_columns:
            print(f"  {logical:22s} -> '{actual}'  OK")
        else:
            cands = suggest(logical, all_columns)
            print(f"  {logical:22s} -> '{actual}'  NOT FOUND"
                  f"   candidates: {cands or '(none found)'}"
                  f"   (set to None in config.py if labels are not stored)")

    print(f"\n{problems} required column mapping(s) need fixing."
          if problems else "\nAll required column mappings resolve. "
          "Proceed to: python 01_table2_rederive.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())