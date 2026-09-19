"""04: per-node-type AST pass + empirical discovery of the original
spaghetti branch-counting rule (supersedes 02's single-count cache).

Pass 1 (resumable, ~5-10 min total): for every function, tally each
control-flow node kind separately (If, For, AsyncFor, While, Try,
TryStar, With, AsyncWith, ExceptHandler, Match, match_case) plus
max_nesting and parse_ok, cached per split.

Pass 2 (seconds): brute-force all candidate rules -- every subset of
optional node kinds on top of the always-included {If, For, While},
crossed with strict (>12) vs inclusive (>=12) thresholds -- against the
stored ap_spaghetti labels on the training split.  The variant with the
fewest mismatches (ideally zero, then verified on validation/test) is
written to outputs/spaghetti_rule.json for downstream scripts (A1, A5,
A13) and for the revised Methods text.

Usage:
    conda activate prime
    python 04_ast_pertype.py
"""
from __future__ import annotations
import ast
import itertools
import json
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

SCRIPT_VERSION = 3

NODE_NAMES = [n for n in config.PERTYPE_NODE_NAMES if hasattr(ast, n)]
NEST_TYPES = tuple(getattr(ast, n) for n in config.NESTING_NODE_NAMES
                   if hasattr(ast, n))

ALWAYS = ["If", "For", "While"]                       # per the manuscript
OPTIONAL = [n for n in NODE_NAMES if n not in ALWAYS]


def parse_body(src: str):
    if not isinstance(src, str) or not src.strip():
        return None
    for attempt in (src, textwrap.dedent(src)):
        try:
            return ast.parse(attempt)
        except (SyntaxError, ValueError, MemoryError):
            continue
    return None


def max_nesting(tree: ast.AST) -> int:
    best = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal best
        for child in ast.iter_child_nodes(node):
            d = depth + 1 if isinstance(child, NEST_TYPES) else depth
            if d > best:
                best = d
            visit(child, d)

    visit(tree, 0)
    return best


def pertype_counts(tree: ast.AST) -> dict[str, int]:
    counts = dict.fromkeys(NODE_NAMES, 0)
    for node in ast.walk(tree):
        name = type(node).__name__
        if name in counts:
            counts[name] += 1
    return counts


def process_split(split: str) -> None:
    path = config.DATA_DIR / config.SPLIT_FILES[split]
    cache = config.AST_CACHE_DIR / f"{split}_pertype"
    cache.mkdir(parents=True, exist_ok=True)
    out_path = config.AST_CACHE_DIR / f"{split}_ast_pertype.parquet"
    if out_path.exists():
        print(f"[{split}] {out_path.name} already consolidated, skipping parse")
        return

    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    body_col = config.COLUMNS["body"]
    print(f"\n[{split}] {total:,} rows (batch={config.AST_BATCH_SIZE:,})")
    t0, row_index, batch_no, done = time.time(), 0, 0, 0

    for batch in pf.iter_batches(batch_size=config.AST_BATCH_SIZE,
                                 columns=[body_col]):
        part = cache / f"part_{batch_no:05d}.parquet"
        n = batch.num_rows
        if not part.exists():
            bodies = batch.column(0).to_pylist()
            recs = []
            for i, src in enumerate(bodies):
                tree = parse_body(src)
                if tree is None:
                    rec = {"row_index": row_index + i, "parse_ok": False,
                           "max_nesting": -1,
                           **dict.fromkeys(NODE_NAMES, -1)}
                else:
                    rec = {"row_index": row_index + i, "parse_ok": True,
                           "max_nesting": max_nesting(tree),
                           **pertype_counts(tree)}
                recs.append(rec)
            pd.DataFrame(recs).to_parquet(part, index=False)
        row_index += n
        batch_no += 1
        done += n
        rate = done / max(time.time() - t0, 1e-9)
        eta = (total - done) / max(rate, 1e-9)
        print(f"  [{split}] {done:,}/{total:,} "
              f"({rate:,.0f} rows/s, ETA {eta/60:.1f} min)", flush=True)

    parts = sorted(cache.glob("part_*.parquet"))
    res = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    res = res.sort_values("row_index").reset_index(drop=True)
    res.to_parquet(out_path, index=False)
    n_fail = int((~res["parse_ok"]).sum())
    print(f"[{split}] consolidated -> {out_path.name}; "
          f"parse failures: {n_fail:,}")


def rule_search() -> None:
    print("\n=== variant search for the original branch-counting rule ===")
    train = pd.read_parquet(config.AST_CACHE_DIR / "train_ast_pertype.parquet")
    stored = pd.read_parquet(
        config.DATA_DIR / config.SPLIT_FILES["train"],
        columns=[config.LABELS["excessive_branching"]]
    )[config.LABELS["excessive_branching"]].astype(bool).to_numpy()

    ok = train["parse_ok"].to_numpy()
    stored_ok = stored[ok]
    base = sum(train[n].to_numpy()[ok] for n in ALWAYS)
    opt_arrays = {n: train[n].to_numpy()[ok] for n in OPTIONAL}

    results = []
    for r in range(len(OPTIONAL) + 1):
        for combo in itertools.combinations(OPTIONAL, r):
            counts = base.copy()
            for n in combo:
                counts = counts + opt_arrays[n]
            for strict in (True, False):
                pred = counts > 12 if strict else counts >= 12
                mism = int((pred != stored_ok).sum())
                results.append((mism, combo, strict))
    results.sort(key=lambda x: x[0])

    print("top 5 variants (mismatches on train, parseable rows):")
    for mism, combo, strict in results[:5]:
        rule = " + ".join(ALWAYS + list(combo))
        thr = "> 12" if strict else ">= 12"
        print(f"  {mism:>8,}  count({rule}) {thr}")

    best_mism, best_combo, best_strict = results[0]
    include = ALWAYS + list(best_combo)

    # verify best variant on validation and test
    verified = {"train": best_mism}
    for split in ("validation", "test"):
        p = config.AST_CACHE_DIR / f"{split}_ast_pertype.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        s = pd.read_parquet(
            config.DATA_DIR / config.SPLIT_FILES[split],
            columns=[config.LABELS["excessive_branching"]]
        )[config.LABELS["excessive_branching"]].astype(bool).to_numpy()
        okm = d["parse_ok"].to_numpy()
        cnt = sum(d[n].to_numpy()[okm] for n in include)
        pred = cnt > 12 if best_strict else cnt >= 12
        verified[split] = int((pred != s[okm]).sum())
    print(f"best variant verified mismatches: {verified}")

    config.OUT_DIR.mkdir(exist_ok=True)
    config.SPAGHETTI_RULE_JSON.write_text(json.dumps({
        "include_node_types": include,
        "strict_greater_than_12": best_strict,
        "mismatches": verified,
        "note": ("Empirically recovered branch-counting rule that "
                 "reproduces the released ap_spaghetti labels; used by "
                 "A1/A5/A13 and to be stated in the revised Methods."),
    }, indent=2))
    print(f"wrote {config.SPAGHETTI_RULE_JSON}")


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"04_ast_pertype v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit(
            "Version mismatch: this script requires config.py v3.")
    splits = sys.argv[1:] or list(config.SPLIT_FILES)
    for split in splits:
        process_split(split)
    rule_search()
    print("\nDone. Paste the variant-search section back into the chat.")


if __name__ == "__main__":
    main()