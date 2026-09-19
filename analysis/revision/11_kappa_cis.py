"""11 (A6): 95% confidence intervals for the tool-agreement kappas.

Computes Cohen's kappa with Fleiss-Cohen-Everitt (1969) analytic 95%
CIs, cross-checked by multinomial parametric bootstrap, for the three
Table 3 comparisons:
  - Long Parameter List vs Pylint R0913   (comparison_dataset.parquet)
  - Excessive Branching  vs Pylint R0912  (comparison_dataset.parquet)
  - High CC              vs radon         (radon_body_agreement_results.csv)

No recomputation of tool output -- everything is derived from the
files your original agreement scripts already produced.

Feeds: R3.20.

Usage:
    conda activate prime
    python 11_kappa_cis.py
    # or with explicit paths:
    python 11_kappa_cis.py /path/to/comparison_dataset.parquet /path/to/radon_body_agreement_results.csv
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

import config

SCRIPT_VERSION = 3

COMPARISON_PARQUET = ("/Users/reemehaidib/PhD_Dataset/PRIME_Output/"
                      "comparison/comparison_dataset.parquet")
RADON_RESULTS_CSV = ("/Users/reemehaidib/PhD_Dataset/PRIME_Output/"
                     "comparison/radon_body_agreement_results.csv")

PYLINT_PAIRS = [
    # (label, stored PRIME column, pylint column)
    ("Long Parameter List vs Pylint R0913", "cs_long_params",
     "pylint_long_params"),
    ("Excessive Branching vs Pylint R0912", "ap_spaghetti",
     "pylint_spaghetti"),
]


def kappa_from_table(a: int, b: int, c: int, d: int) -> float:
    """2x2 cells: a=both+, b=PRIME+ only, c=tool+ only, d=both-."""
    N = a + b + c + d
    po = (a + d) / N
    p1, p2 = (a + b) / N, (a + c) / N
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe)


def kappa_se_fce(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Fleiss-Cohen-Everitt large-sample kappa SE for a 2x2 table."""
    N = a + b + c + d
    p11, p12, p21, p22 = a / N, b / N, c / N, d / N
    r = np.array([p11 + p12, p21 + p22])       # PRIME marginals (+,-)
    s = np.array([p11 + p21, p12 + p22])       # tool marginals (+,-)
    po = p11 + p22
    pe = float(r @ s)
    k = (po - pe) / (1 - pe)
    t1 = (p11 * ((1 - pe) - (r[0] + s[0]) * (1 - po)) ** 2
          + p22 * ((1 - pe) - (r[1] + s[1]) * (1 - po)) ** 2)
    t2 = (1 - po) ** 2 * (p12 * (s[0] + r[1]) ** 2
                          + p21 * (s[1] + r[0]) ** 2)
    t3 = (po * pe - 2 * pe + po) ** 2
    var = (t1 + t2 - t3) / (N * (1 - pe) ** 4)
    return k, float(np.sqrt(var))


def boot_ci(a: int, b: int, c: int, d: int,
            rng: np.random.Generator) -> tuple[float, float]:
    N = a + b + c + d
    probs = np.array([a, b, c, d]) / N
    ks = np.empty(config.N_BOOTSTRAP)
    for i in range(config.N_BOOTSTRAP):
        aa, bb, cc, dd = rng.multinomial(N, probs)
        ks[i] = kappa_from_table(aa, bb, cc, dd)
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def report(name: str, a: int, b: int, c: int, d: int,
           rng: np.random.Generator) -> dict:
    k, se = kappa_se_fce(a, b, c, d)
    lo, hi = k - 1.96 * se, k + 1.96 * se
    blo, bhi = boot_ci(a, b, c, d, rng)
    print(f"  {name}")
    print(f"    cells: both+={a:,}  PRIME+only={b:,}  tool+only={c:,} "
          f"both-={d:,}  (N={a+b+c+d:,})")
    print(f"    kappa = {k:.3f}  analytic 95% CI [{lo:.3f}, {hi:.3f}]"
          f"  bootstrap [{blo:.3f}, {bhi:.3f}]")
    return {"comparison": name, "both_pos": a, "prime_only": b,
            "tool_only": c, "both_neg": d, "kappa": round(k, 4),
            "se": round(se, 5), "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "boot_ci_low": round(blo, 4), "boot_ci_high": round(bhi, 4)}


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"11_kappa_cis v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    comp_path = sys.argv[1] if len(sys.argv) > 1 else COMPARISON_PARQUET
    radon_path = sys.argv[2] if len(sys.argv) > 2 else RADON_RESULTS_CSV
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    rows = []

    # ---- Pylint pairs from the per-function comparison dataset -------
    print(f"loading {comp_path} ...")
    needed = {c for _, pc, tc in PYLINT_PAIRS for c in (pc, tc)}
    comp = pd.read_parquet(comp_path)
    missing = needed - set(comp.columns)
    if missing:
        raise SystemExit(f"columns missing from comparison dataset: "
                         f"{missing}\navailable: {list(comp.columns)}")
    print(f"  rows: {len(comp):,}")
    print("\ntool-agreement kappas with 95% CIs:")
    for name, prime_col, tool_col in PYLINT_PAIRS:
        p = comp[prime_col].astype(bool).to_numpy()
        t = comp[tool_col].astype(bool).to_numpy()
        a = int((p & t).sum()); b = int((p & ~t).sum())
        c = int((~p & t).sum()); d = int((~p & ~t).sum())
        rows.append(report(name, a, b, c, d, rng))

    # ---- radon High CC from the saved results table ------------------
    r = pd.read_csv(radon_path).iloc[0]
    # NOTE column-name convention in that file: 'FP' = PRIME+/tool-,
    # 'FN' = PRIME-/tool+ (swapped vs sklearn naming; metrics correct).
    rows.append(report("High CC vs radon cc_visit",
                       int(r["TP"]), int(r["FP"]), int(r["FN"]),
                       int(r["TN"]), rng))

    config.OUT_DIR.mkdir(exist_ok=True)
    out_csv = config.OUT_DIR / "tool_kappa_cis.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("Done. Paste the printed block back into the chat.")


if __name__ == "__main__":
    main()