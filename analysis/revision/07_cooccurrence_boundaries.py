"""07 (A4 + A5): pairwise co-occurrence and threshold boundary counts.

A4: whole-corpus 5x5 subtype co-occurrence -- counts, Jaccard, lift --
    answering R1's "which smells occur together" (R1-1.13).
A5: number of functions sitting exactly AT each threshold (NLOC = 14,
    params = 5, CC = 10, branches = 12, calls = 15), per split and
    corpus, for the strict-inequality specification (R3.18).

Usage:
    conda activate prime
    python 07_cooccurrence_boundaries.py    (~1-2 min with caches)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config

SCRIPT_VERSION = 3
SUBTYPES = list(config.THRESHOLDS.keys())


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"07_cooccurrence_boundaries v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    config.OUT_DIR.mkdir(exist_ok=True)

    masks_all: dict[str, list] = {s: [] for s in SUBTYPES}
    boundary_rows = []
    for split in config.SPLIT_FILES:
        print(f"loading {split} ...")
        _proj, metrics, masks = common.load_masks(split)
        for s in SUBTYPES:
            masks_all[s].append(masks[s])
        row = {"split": split}
        for s, (metric, t) in config.THRESHOLDS.items():
            row[s] = int((metrics[metric] == t).sum())
        boundary_rows.append(row)

    M = {s: np.concatenate(v) for s, v in masks_all.items()}
    N = len(next(iter(M.values())))

    # ------------------------------ A5: boundary counts ---------------
    corpus_row = {"split": "corpus"}
    for s in SUBTYPES:
        corpus_row[s] = sum(r[s] for r in boundary_rows)
    boundary_rows.append(corpus_row)
    btab = pd.DataFrame(boundary_rows)
    btab.to_csv(config.OUT_DIR / "boundary_counts.csv", index=False)
    print("\nfunctions exactly AT each threshold (excluded by strict '>'):")
    for r in btab.itertuples():
        parts = [f"{s}={getattr(r, s):,}" for s in SUBTYPES]
        print(f"  [{r.split:10s}] " + "  ".join(parts))

    # ------------------------------ A4: pairwise ----------------------
    n = {s: int(M[s].sum()) for s in SUBTYPES}
    rows = []
    print(f"\npairwise co-occurrence (corpus, N = {N:,}):")
    print(f"  {'pair':48s} {'n_both':>9s} {'jaccard':>8s} {'lift':>7s}")
    for i, a in enumerate(SUBTYPES):
        for b in SUBTYPES[i + 1:]:
            nb = int((M[a] & M[b]).sum())
            union = n[a] + n[b] - nb
            jac = nb / union if union else 0.0
            lift = (nb * N) / (n[a] * n[b]) if n[a] and n[b] else 0.0
            rows.append({"subtype_a": a, "subtype_b": b,
                         "n_a": n[a], "n_b": n[b], "n_both": nb,
                         "jaccard": round(jac, 4), "lift": round(lift, 2)})
            print(f"  {a} x {b:<28s} {nb:>9,} {jac:>8.4f} {lift:>7.2f}")
    pd.DataFrame(rows).to_csv(
        config.OUT_DIR / "pairwise_cooccurrence.csv", index=False)
    print(f"\nwrote {config.OUT_DIR / 'pairwise_cooccurrence.csv'}")
    print("wrote " + str(config.OUT_DIR / "boundary_counts.csv"))
    print("\nDone. Paste both printed blocks back into the chat.")


if __name__ == "__main__":
    main()