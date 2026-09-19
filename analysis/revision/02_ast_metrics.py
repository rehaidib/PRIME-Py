"""02: full-corpus AST pass over function_body (serves A1, A5, A13).

Computes, for every function in every split:
  - branch_count : branch-inducing statements (if/for/while/try/with/
                   except, incl. async variants) -- needed because the
                   dataset stores no branch-count column
  - max_nesting  : maximum nesting depth of control-flow statements
  - parse_ok     : whether the body parsed

Results are cached as one Parquet part per batch under
ast_cache/<split>/part_XXXXX.parquet (columns: row_index, branch_count,
max_nesting, parse_ok).  Re-running skips completed parts, so the job is
safely interruptible.  Row order in Parquet is stable, so row_index
aligns positionally with the split file for later joins (A1/A5/A13).

After each split completes, verifies the stored ap_spaghetti label
against (branch_count > 12) on parseable rows.  High disagreement means
our counting rule differs from the original labeling script -- in that
case, paste the original script's branch-counting snippet into the chat
and BRANCH_NODE_NAMES in config.py gets adjusted to mirror it, then
delete ast_cache/ and re-run.

Usage:
    conda activate prime
    python 02_ast_metrics.py            # all splits
    python 02_ast_metrics.py train      # one split
Runtime: roughly 15-45 min for the full corpus on a laptop.
"""
from __future__ import annotations
import ast
import sys
import textwrap
import time

import pandas as pd
import pyarrow.parquet as pq

import config

BRANCH_TYPES = tuple(
    getattr(ast, n) for n in config.BRANCH_NODE_NAMES if hasattr(ast, n)
)
NEST_TYPES = tuple(
    getattr(ast, n) for n in config.NESTING_NODE_NAMES if hasattr(ast, n)
)


def parse_body(src: str):
    """Parse a stored function body, tolerating leading indentation."""
    if not isinstance(src, str) or not src.strip():
        return None
    for attempt in (src, textwrap.dedent(src)):
        try:
            return ast.parse(attempt)
        except (SyntaxError, ValueError, MemoryError):
            continue
    return None


def branch_count(tree: ast.AST) -> int:
    return sum(isinstance(node, BRANCH_TYPES) for node in ast.walk(tree))


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


def process_split(split: str) -> None:
    path = config.DATA_DIR / config.SPLIT_FILES[split]
    cache = config.AST_CACHE_DIR / split
    cache.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    body_col = config.COLUMNS["body"]

    print(f"\n[{split}] {total:,} rows "
          f"(batch={config.AST_BATCH_SIZE:,}, cache={cache})")
    t0 = time.time()
    row_index = 0
    batch_no = 0
    done_rows = 0

    for batch in pf.iter_batches(batch_size=config.AST_BATCH_SIZE,
                                 columns=[body_col]):
        part = cache / f"part_{batch_no:05d}.parquet"
        n = batch.num_rows
        if part.exists():
            row_index += n
            batch_no += 1
            done_rows += n
            continue

        bodies = batch.column(0).to_pylist()
        recs = []
        for i, src in enumerate(bodies):
            tree = parse_body(src)
            if tree is None:
                recs.append((row_index + i, -1, -1, False))
            else:
                recs.append((row_index + i, branch_count(tree),
                             max_nesting(tree), True))
        pd.DataFrame(recs, columns=["row_index", "branch_count",
                                    "max_nesting", "parse_ok"]
                     ).to_parquet(part, index=False)

        row_index += n
        batch_no += 1
        done_rows += n
        rate = done_rows / max(time.time() - t0, 1e-9)
        eta = (total - done_rows) / max(rate, 1e-9)
        print(f"  [{split}] {done_rows:,}/{total:,} rows "
              f"({rate:,.0f} rows/s, ETA {eta/60:.1f} min)", flush=True)

    # ------------------- consolidate + verify -------------------------
    parts = sorted(cache.glob("part_*.parquet"))
    res = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    res = res.sort_values("row_index").reset_index(drop=True)
    out = config.AST_CACHE_DIR / f"{split}_ast_metrics.parquet"
    res.to_parquet(out, index=False)

    n_fail = int((~res["parse_ok"]).sum())
    print(f"[{split}] consolidated -> {out}")
    print(f"[{split}] parse failures: {n_fail:,} "
          f"({100 * n_fail / len(res):.2f}%)")

    stored = pd.read_parquet(path, columns=[config.LABELS["excessive_branching"]]
                             )[config.LABELS["excessive_branching"]].astype(bool)
    ok = res["parse_ok"].to_numpy()
    recomputed = (res["branch_count"].to_numpy() > 12)
    agree = (recomputed[ok] == stored.to_numpy()[ok])
    n_ok = int(ok.sum())
    n_dis = int(n_ok - agree.sum())
    print(f"[{split}] ap_spaghetti vs (branch_count > 12) on parseable rows: "
          f"{n_dis:,} / {n_ok:,} disagree ({100 * n_dis / n_ok:.3f}%)")
    if n_dis / max(n_ok, 1) > 0.005:
        print(f"[{split}] !! counting-rule mismatch likely -- paste the "
              "original labeling script's branch-counting code into the chat "
              "so BRANCH_NODE_NAMES can be mirrored exactly, then delete "
              "ast_cache/ and re-run.")


def main() -> None:
    splits = sys.argv[1:] or list(config.SPLIT_FILES)
    for split in splits:
        if split not in config.SPLIT_FILES:
            raise SystemExit(f"unknown split '{split}'")
        process_split(split)
    print("\nDone. Paste the parse-failure and verification lines "
          "back into the chat.")


if __name__ == "__main__":
    main()