"""10 (A2): direct label-level evaluation on the Sandouka & Aljamaan
benchmark (redesigned RQ4).

Applies the PRIME-Py Long Method detection strategy to the 894 released
benchmark instances and evaluates against the expert labels:
  - PRIMARY: corpus-relative percentile rules (P70/P75/P80/P90 of the
    benchmark's own scloc distribution; thresholds are estimated from
    the size distribution only, with no access to labels).
  - SECONDARY: the fixed literal cutoff (scloc > 14), expected to
    transfer poorly (scloc != Lizard NLOC; corpus-specific values).
  - Context: AUROC of scloc as a score, and the best-F1 threshold by
    exhaustive search.
The P75 kappa gets a 10,000-rep bootstrap 95% CI with the threshold
re-estimated inside every resample.

Feeds: R2.9, R3.25 (Reviewer 3 priority 4).

Usage:
    conda activate prime
    python 10_sandouka_direct_eval.py /path/to/Python_LongMethodSmell_Dataset.csv
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

import config

SCRIPT_VERSION = 3
LABEL_COL = "Experince Based"   # note: source file has a trailing space


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged_pct": round(100 * pred.mean(), 1),
        "precision": round(prec, 3), "recall": round(rec, 3),
        "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "accuracy": round((tp + tn) / len(y), 3),
        "kappa": round(kappa(pred.astype(int), y), 3),
    }


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"10_sandouka_direct_eval v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    if len(sys.argv) != 2:
        raise SystemExit("usage: python 10_sandouka_direct_eval.py "
                         "/path/to/Python_LongMethodSmell_Dataset.csv")

    df = pd.read_csv(sys.argv[1])
    df.columns = [c.strip() for c in df.columns]
    y = df[LABEL_COL].astype(int).to_numpy()
    scloc = df["scloc"].astype(float).to_numpy()
    print(f"N = {len(df)}, positives = {y.sum()} ({100 * y.mean():.1f}%)")

    rows = []
    for p in [70, 75, 80, 90]:
        t = float(np.percentile(scloc, p))
        m = metrics((scloc > t).astype(int), y)
        rows.append({"rule": f"P{p} (scloc > {t:.0f})",
                     "threshold": t, **m})
        print(f"  P{p}: threshold={t:.0f}  kappa={m['kappa']}  "
              f"F1={m['f1']}  Prec={m['precision']}  Rec={m['recall']}")

    m14 = metrics((scloc > 14).astype(int), y)
    rows.append({"rule": "fixed (scloc > 14)", "threshold": 14.0, **m14})
    print(f"  fixed > 14: kappa={m14['kappa']}  F1={m14['f1']}  "
          f"flagged={m14['flagged_pct']}%")

    # bootstrap CI for the P75 rule, threshold re-estimated per resample
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    ks = np.empty(config.N_BOOTSTRAP)
    for i in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        yb, sb = y[idx], scloc[idx]
        tb = np.percentile(sb, 75)
        ks[i] = kappa((sb > tb).astype(int), yb)
    lo, hi = np.percentile(ks, [2.5, 97.5])
    print(f"  P75 kappa bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")

    # threshold-free context
    ranks = pd.Series(scloc).rank(method="average").to_numpy()
    n1 = int(y.sum())
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    best_f1, best_t = 0.0, None
    for t in np.unique(scloc):
        m = metrics((scloc > t).astype(int), y)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    print(f"  AUROC(scloc) = {auc:.3f}; best-F1 threshold: "
          f"> {best_t:.0f} (F1 = {best_f1:.3f})")

    out = pd.DataFrame(rows)
    out.loc[out["rule"].str.startswith("P75"), "kappa_ci95_low"] = round(lo, 3)
    out.loc[out["rule"].str.startswith("P75"), "kappa_ci95_high"] = round(hi, 3)
    config.OUT_DIR.mkdir(exist_ok=True)
    out_csv = config.OUT_DIR / "sandouka_direct_eval.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()