"""06 (A3): repository-level variation and cluster-aware uncertainty.

Builds per-project counts for all five subtypes and the four category
aggregates, then:
  - project-level (cluster) bootstrap 95% CIs for every prevalence
    estimate, per split and for the whole corpus (resampling projects,
    not functions);
  - per-project prevalence distributions (CSV + median/IQR summary)
    for the revised "Repository-Level Variation" section.

Feeds: E5, R3.23, R3.24.

Usage:
    conda activate prime
    python 06_cluster_bootstrap.py       (~2-3 min)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config

SCRIPT_VERSION = 3

MEASURES = ["long_method", "high_cc", "long_parameter_list",
            "any_code_smell", "excessive_branching", "high_fan_out",
            "any_scs", "both", "neither"]


def per_project_counts(split: str) -> pd.DataFrame:
    proj, _metrics, masks = common.load_masks(split)
    df = pd.DataFrame({"project": proj})
    for m in MEASURES:
        df[m] = masks[m]
    g = df.groupby("project", sort=False)
    out = g[MEASURES].sum()
    out["n_functions"] = g.size()
    out["split"] = split
    return out.reset_index()


def boot_cis(tab: pd.DataFrame, rng: np.random.Generator) -> dict:
    P = len(tab)
    tot = tab["n_functions"].to_numpy(dtype=np.float64)
    pos = {m: tab[m].to_numpy(dtype=np.float64) for m in MEASURES}
    sums = {m: np.empty(config.N_BOOTSTRAP) for m in MEASURES}
    denom = np.empty(config.N_BOOTSTRAP)
    block = 1000
    for start in range(0, config.N_BOOTSTRAP, block):
        b = min(block, config.N_BOOTSTRAP - start)
        idx = rng.integers(0, P, size=(b, P))
        denom[start:start + b] = tot[idx].sum(axis=1)
        for m in MEASURES:
            sums[m][start:start + b] = pos[m][idx].sum(axis=1)
    out = {}
    for m in MEASURES:
        prev = 100 * sums[m] / denom
        out[m] = (float(100 * pos[m].sum() / tot.sum()),
                  float(np.percentile(prev, 2.5)),
                  float(np.percentile(prev, 97.5)))
    return out


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"06_cluster_bootstrap v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    config.OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)

    tabs = []
    for split in config.SPLIT_FILES:
        print(f"building per-project counts [{split}] ...")
        tabs.append(per_project_counts(split))
    allp = pd.concat(tabs, ignore_index=True)
    allp.to_csv(config.OUT_DIR / "per_project_counts.csv", index=False)
    print(f"projects: {len(allp):,} (expected 2,797)")

    ci_rows = []
    scopes = {s: t for s, t in zip(config.SPLIT_FILES, tabs)}
    scopes["corpus"] = allp
    for scope, tab in scopes.items():
        print(f"bootstrapping [{scope}] ({len(tab):,} projects, "
              f"{config.N_BOOTSTRAP:,} reps) ...")
        for m, (prev, lo, hi) in boot_cis(tab, rng).items():
            ci_rows.append({"scope": scope, "measure": m,
                            "prevalence_pct": round(prev, 2),
                            "ci95_low": round(lo, 2),
                            "ci95_high": round(hi, 2)})
    cis = pd.DataFrame(ci_rows)
    cis.to_csv(config.OUT_DIR / "prevalence_cluster_cis.csv", index=False)
    print(f"wrote {config.OUT_DIR / 'prevalence_cluster_cis.csv'}")

    # ---- per-project prevalence distributions ------------------------
    prev = allp.copy()
    for m in MEASURES:
        prev[m] = 100 * prev[m] / prev["n_functions"]
    prev[["project", "split", "n_functions"] + MEASURES].to_csv(
        config.OUT_DIR / "per_project_prevalence.csv", index=False)

    print("\nper-project prevalence distribution (corpus, % of functions):")
    for m in MEASURES:
        q = np.percentile(prev[m], [25, 50, 75])
        print(f"  {m:22s} median={q[1]:6.2f}  IQR=[{q[0]:6.2f}, {q[2]:6.2f}]")

    print("\ncorpus prevalence with cluster-bootstrap 95% CIs:")
    for r in cis[cis["scope"] == "corpus"].itertuples():
        print(f"  {r.measure:22s} {r.prevalence_pct:6.2f}% "
              f"[{r.ci95_low:6.2f}, {r.ci95_high:6.2f}]")

    print("\nDone. Paste both printed blocks back into the chat.")


if __name__ == "__main__":
    main()