"""05 (A1): threshold sensitivity analysis, whole corpus.

For each subtype, sweeps its threshold over the configured grid while
holding the other four at canonical values, reporting the subtype's
prevalence plus the category aggregates (ANY code smell, ANY SCS,
co-occurrence, neither) at every setting.  Long Method additionally
sweeps training-split percentiles (P70/P75/P80/P90 -- P75 should
reproduce the canonical 14) and fixed literature-style cutoffs.

Feeds: E5, R1-1.8, R2.5, R3.19.

Usage:
    conda activate prime
    python 05_sensitivity.py            (~2-4 min; caches make reruns fast)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config

SCRIPT_VERSION = 3


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"05_sensitivity v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    config.OUT_DIR.mkdir(exist_ok=True)

    metrics_all: dict[str, list] = {k: [] for k in
                                    ["nloc", "cc", "calls", "params",
                                     "branches"]}
    train_nloc = None
    for split in config.SPLIT_FILES:
        print(f"loading {split} ...")
        _proj, metrics, _masks = common.load_masks(split)
        for k in metrics_all:
            metrics_all[k].append(metrics[k])
        if split == "train":
            train_nloc = metrics["nloc"]
    M = {k: np.concatenate(v) for k, v in metrics_all.items()}
    N = len(M["nloc"])
    print(f"corpus rows: {N:,}")

    canonical = {s: M[m] > t for s, (m, t) in config.THRESHOLDS.items()}

    def aggregates(masks: dict) -> dict:
        any_cs = np.logical_or.reduce(
            [masks[s] for s in config.CODE_SMELL_SUBTYPES])
        any_scs = np.logical_or.reduce(
            [masks[s] for s in config.SCS_SUBTYPES])
        both = any_cs & any_scs
        return {
            "any_code_smell_pct": 100 * any_cs.mean(),
            "any_scs_pct":        100 * any_scs.mean(),
            "cooccurrence_pct":   100 * both.mean(),
            "neither_pct":        100 * (~any_cs & ~any_scs).mean(),
        }

    rows = []

    def add_row(subtype: str, label: str, t_value: float,
                mask: np.ndarray, is_canonical: bool) -> None:
        masks = dict(canonical)
        masks[subtype] = mask
        rows.append({
            "subtype": subtype, "threshold_label": label,
            "threshold_value": t_value,
            "canonical": is_canonical,
            "prevalence_pct": round(100 * mask.mean(), 2),
            **{k: round(v, 2) for k, v in aggregates(masks).items()},
        })

    # ---- Long Method: percentile strategy + fixed cutoffs -----------
    print("\nLong Method training-split percentiles:")
    for p in config.SENSITIVITY_GRIDS["long_method_percentiles"]:
        t = float(np.percentile(train_nloc, p))
        print(f"  P{p}(train NLOC) = {t}")
        add_row("long_method", f"P{p} (train)", t,
                M["nloc"] > t, p == 75)
    for t in config.SENSITIVITY_GRIDS["long_method_fixed_nloc"]:
        add_row("long_method", f"fixed > {t}", t, M["nloc"] > t, False)

    # ---- the four fixed-threshold subtypes --------------------------
    grid_map = {
        "high_cc": ("cc", config.SENSITIVITY_GRIDS["high_cc"]),
        "long_parameter_list": ("params",
                                config.SENSITIVITY_GRIDS["long_parameter_list"]),
        "excessive_branching": ("branches",
                                config.SENSITIVITY_GRIDS["excessive_branching"]),
        "high_fan_out": ("calls", config.SENSITIVITY_GRIDS["high_fan_out"]),
    }
    for subtype, (metric, grid) in grid_map.items():
        t_canon = config.THRESHOLDS[subtype][1]
        for t in grid:
            add_row(subtype, f"> {t}", t, M[metric] > t, t == t_canon)

    out = pd.DataFrame(rows)
    out_csv = config.OUT_DIR / "sensitivity_results.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    print("\nprevalence by threshold (canonical rows marked *):")
    for subtype in out["subtype"].unique():
        sub = out[out["subtype"] == subtype]
        parts = [f"{r.threshold_label}{'*' if r.canonical else ''}: "
                 f"{r.prevalence_pct}%" for r in sub.itertuples()]
        print(f"  {subtype:22s} " + " | ".join(parts))

    print("\nDone. Paste the printed summary back into the chat.")


if __name__ == "__main__":
    main()