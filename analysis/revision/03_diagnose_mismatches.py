"""03: diagnose the three findings from the 01/02 runs.

Part A -- Long Parameter List rule:
    Tests whether cs_long_params was computed as (non-self params > 5),
    deriving the non-self count two ways:
      H1: num_parameter minus 1 when the first entry of function_params
          is 'self' or 'cls'
      H2: len(function_params) minus self/cls, ignoring num_parameter
    Reports mismatch counts per hypothesis per split; a hypothesis that
    reaches 0 everywhere is the original rule.

Part B -- code_smell_label composition:
    Re-checks the stored aggregate against the union of the three
    stored subtype columns (independent of any recomputation), with
    direction breakdown.  Any stored-True/union-False rows are genuine
    dataset inconsistencies (Zenodo v2 changelog material).

Part C -- ap_spaghetti counting-rule forensics (train split):
    For rows where (recomputed branch_count > 12) disagrees with the
    stored label: direction counts, branch-count histograms near the
    threshold, syntax-marker frequencies (async / except* / match), and
    two truncated example bodies per direction.

Usage:
    conda activate prime
    python 03_diagnose_mismatches.py
"""
from __future__ import annotations
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

SCRIPT_VERSION = 2


# ------------------------------------------------------------ helpers
def parse_param_tokens(val) -> list[str]:
    """function_params -> list of bare parameter names.

    Handles: real list/array types, "['self', 'x']", "(self, x)",
    "self, x", annotations ("x: int") and defaults ("x=1")."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        raw = [str(v) for v in val]
    else:
        t = str(val).strip()
        if t in ("", "[]", "()", "nan", "None"):
            return []
        if t[0] in "[(":
            t = t[1:]
        if t and t[-1] in "])":
            t = t[:-1]
        raw = t.split(",")
    toks = []
    for r in raw:
        tok = r.strip().strip("'\"").strip()
        tok = tok.split(":")[0].split("=")[0].strip().lstrip("*")
        if tok:
            toks.append(tok)
    return toks


def fetch_bodies(split: str, indices: set[int]) -> dict[int, str]:
    """Fetch function bodies for positional row indices in one pass."""
    path = config.DATA_DIR / config.SPLIT_FILES[split]
    pf = pq.ParquetFile(path)
    out: dict[int, str] = {}
    base = 0
    for batch in pf.iter_batches(batch_size=config.AST_BATCH_SIZE,
                                 columns=[config.COLUMNS["body"]]):
        hits = [i for i in indices if base <= i < base + batch.num_rows]
        if hits:
            col = batch.column(0)
            for i in hits:
                out[i] = col[i - base].as_py()
        base += batch.num_rows
        if len(out) == len(indices):
            break
    return out


# -------------------------------------------------------------- parts
def part_a_and_b(split: str) -> None:
    path = config.DATA_DIR / config.SPLIT_FILES[split]
    cols = [config.COLUMNS["params"], "function_params", "class_name",
            config.LABELS["long_parameter_list"],
            config.LABELS["long_method"], config.LABELS["high_cc"],
            config.AGGREGATE_LABELS["code_smell"]]
    df = pd.read_parquet(path, columns=cols)

    stored = df[config.LABELS["long_parameter_list"]].astype(bool).to_numpy()
    nump = df[config.COLUMNS["params"]].to_numpy()

    if split == "train":
        print("\n  sample function_params values (raw):")
        for v in df["function_params"].head(5):
            print(f"    {v!r}")

    tokens = df["function_params"].map(parse_param_tokens)
    first = tokens.map(lambda t: t[0] if t else "")
    is_selfcls = first.isin(["self", "cls"]).to_numpy()
    nlist = tokens.map(len).to_numpy()

    h0 = nump > 5
    h1 = (nump - is_selfcls.astype(int)) > 5
    h2 = (nlist - is_selfcls.astype(int)) > 5

    print(f"\n  [A:{split}] LPL rule hypotheses vs stored cs_long_params:")
    for name, pred in [("H0 num_parameter > 5 (naive)", h0),
                       ("H1 (num_parameter - self/cls) > 5", h1),
                       ("H2 (len(function_params) - self/cls) > 5", h2)]:
        mism = int((pred != stored).sum())
        print(f"    {name:42s}: {mism:,} mismatches"
              f"{'   <-- MATCHES' if mism == 0 else ''}")

    same_count = int((nump == nlist).sum())
    print(f"    num_parameter == len(function_params) on "
          f"{same_count:,}/{len(df):,} rows")

    boundary = h0 & ~stored & is_selfcls & (nump == 6)
    h0_extra = int((h0 & ~stored).sum())
    print(f"    of {h0_extra:,} naive-only positives, "
          f"{int(boundary.sum()):,} are self/cls methods with exactly "
          f"6 raw params (boundary-flip explanation)")

    # -------- Part B: stored aggregate vs stored union ---------------
    union = (df[config.LABELS["long_method"]].astype(bool)
             | df[config.LABELS["high_cc"]].astype(bool)
             | df[config.LABELS["long_parameter_list"]].astype(bool)
             ).to_numpy()
    agg = df[config.AGGREGATE_LABELS["code_smell"]].astype(bool).to_numpy()
    agg_only = int((agg & ~union).sum())
    union_only = int((~agg & union).sum())
    print(f"  [B:{split}] code_smell_label vs union(stored cs_*): "
          f"aggregate-only={agg_only:,}, union-only={union_only:,} "
          f"{'OK' if agg_only == union_only == 0 else '!! dataset inconsistency'}")
    print(f"    sums: aggregate={int(agg.sum()):,}, "
          f"union={int(union.sum()):,}")


def part_c(split: str = "train") -> None:
    path = config.DATA_DIR / config.SPLIT_FILES[split]
    ast_path = config.AST_CACHE_DIR / f"{split}_ast_metrics.parquet"
    if not ast_path.exists():
        print(f"\n  [C] {ast_path} missing -- run 02 first; skipping")
        return
    res = pd.read_parquet(ast_path).sort_values("row_index")
    stored = pd.read_parquet(
        path, columns=[config.LABELS["excessive_branching"]]
    )[config.LABELS["excessive_branching"]].astype(bool).to_numpy()

    bc = res["branch_count"].to_numpy()
    ok = res["parse_ok"].to_numpy()
    rec = bc > 12

    rec_only = np.where(ok & rec & ~stored)[0]     # we say yes, label no
    sto_only = np.where(ok & ~rec & stored)[0]     # label yes, we say no
    print(f"\n  [C:{split}] spaghetti disagreement directions: "
          f"recomputed-only={len(rec_only):,}, stored-only={len(sto_only):,}")

    for name, idx in [("recomputed-only", rec_only), ("stored-only", sto_only)]:
        if len(idx) == 0:
            continue
        hist = Counter(bc[idx]).most_common(6)
        print(f"    {name} branch_count histogram (top): {hist}")

    sample = list(rec_only[:400]) + list(sto_only[:400])
    if not sample:
        print("    no disagreements to sample")
        return
    bodies = fetch_bodies(split, set(int(i) for i in sample))
    markers = ["async for", "async with", "except*", "match ", "case ",
               "elif ", "assert ", "comprehension_if"]
    counts = Counter()
    for i, b in bodies.items():
        if not isinstance(b, str):
            continue
        for m in markers[:-1]:
            if m in b:
                counts[m] += 1
        if " if " in b and ("for " in b) and ("[" in b or "{" in b):
            counts["comprehension_if?"] += 1
    print(f"    syntax markers in {len(bodies):,} disagreeing bodies: "
          f"{dict(counts)}")

    for name, idx in [("recomputed-only", rec_only), ("stored-only", sto_only)]:
        for i in idx[:2]:
            b = bodies.get(int(i), "")
            lines = (b or "").splitlines()[:25]
            print(f"\n    --- example [{name}] row {int(i)}: "
                  f"recomputed branch_count={int(bc[i])}, "
                  f"stored={bool(stored[i])} ---")
            for ln in lines:
                print(f"      {ln[:110]}")
            if b and len(b.splitlines()) > 25:
                print("      ... (truncated)")


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"03_diagnose_mismatches v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: replace stale files.")
    for split in config.SPLIT_FILES:
        part_a_and_b(split)
    part_c("train")
    print("\nDone. Paste the full output back into the chat.")


if __name__ == "__main__":
    main()