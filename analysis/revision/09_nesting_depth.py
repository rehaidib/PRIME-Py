"""09 (A13): nesting-depth evidence for the Excessive Branching label.

Compares maximum control-flow nesting depth between Excessive
Branching positives and the rest of the corpus, using the max_nesting
values already in the pertype cache.  Reports medians, IQRs, the
proportion of deeply nested functions (depth >= 3 and >= 4), and
Cliff's delta as a rank-based effect size.

This is EVIDENCE ONLY -- the detection rule is unchanged.  It supports
the Discussion sentence connecting the Excessive Branching label to
the tangled-control-flow characterization of Spaghetti Code.

Usage:
    conda activate prime
    python 09_nesting_depth.py          (~1 min)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config

SCRIPT_VERSION = 3


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based Cliff's delta between groups x and y (ties averaged)."""
    combined = pd.Series(np.concatenate([x, y]))
    ranks = combined.rank(method="average").to_numpy()
    r_x = ranks[: len(x)].sum()
    u_x = r_x - len(x) * (len(x) + 1) / 2
    return float(2.0 * u_x / (len(x) * len(y)) - 1.0)


def describe(v: np.ndarray) -> dict:
    q = np.percentile(v, [25, 50, 75])
    return {"n": len(v), "median": float(q[1]),
            "iqr_low": float(q[0]), "iqr_high": float(q[2]),
            "mean": float(v.mean()),
            "pct_ge3": round(100 * (v >= 3).mean(), 2),
            "pct_ge4": round(100 * (v >= 4).mean(), 2)}


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"09_nesting_depth v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    config.OUT_DIR.mkdir(exist_ok=True)

    nest_parts, eb_parts = [], []
    for split in config.SPLIT_FILES:
        print(f"loading {split} ...")
        nest_parts.append(common.max_nesting(split))
        eb_parts.append(common.branch_counts(split) > 12)
    nest = np.concatenate(nest_parts)
    eb = np.concatenate(eb_parts)

    groups = {
        "excessive_branching_positives": nest[eb],
        "all_other_functions":           nest[~eb],
        "whole_corpus":                  nest,
    }
    rows = []
    print("\nmax nesting depth by group:")
    for name, v in groups.items():
        d = describe(v)
        rows.append({"group": name, **d})
        print(f"  {name:30s} n={d['n']:>11,}  median={d['median']:.0f}  "
              f"IQR=[{d['iqr_low']:.0f}, {d['iqr_high']:.0f}]  "
              f"mean={d['mean']:.2f}  >=3: {d['pct_ge3']}%  "
              f">=4: {d['pct_ge4']}%")

    delta = cliffs_delta(groups["excessive_branching_positives"],
                         groups["all_other_functions"])
    print(f"\nCliff's delta (EB positives vs rest): {delta:.3f}")
    print("  interpretation: |d|<0.147 negligible, <0.33 small, "
          "<0.474 medium, else large (Romano et al., 2006)")

    out = pd.DataFrame(rows)
    out["cliffs_delta_vs_rest"] = [delta, None, None]
    out_csv = config.OUT_DIR / "nesting_depth_evidence.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\nDone. Paste the printed block back into the chat.")


if __name__ == "__main__":
    main()