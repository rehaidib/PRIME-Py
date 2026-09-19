"""13: regenerate Figure 2 as a two-panel co-occurrence figure.

Panel A: the four mutually exclusive populations (whole corpus).
Panel B: UpSet-style plot of sub-type combinations among affected
functions: one bar per observed combination of the five sub-type
flags, ranked by frequency, with a membership dot matrix beneath.

Outputs figure2_cooccurrence_v2.pdf / .png and a combinations CSV,
and prints the combination table so exact values can be cited.

Usage:
    conda activate prime
    python 13_figure2_combinations.py
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SCRIPT_VERSION = 3
TOP_K = 12

FLAGS = [("LM", "cs_long_method"), ("HCC", "cs_high_cc"),
         ("LPL", "cs_long_params"), ("EB", "ap_spaghetti"),
         ("HFO", "ap_high_fanout")]
FLAG_NAMES = {"LM": "Long Method", "HCC": "High CC",
              "LPL": "Long Param. List", "EB": "Excessive Branching",
              "HFO": "High Fan-Out"}

DATA_DIR = (getattr(config, "LABELLED_DIR", None)
            or getattr(config, "DATA_DIR", None)
            or "/Users/reemehaidib/PhD_Dataset/PRIME_Output/labelled")


def load_flags() -> pd.DataFrame:
    cols = [c for _, c in FLAGS]
    parts = []
    for name in ["train", "validation", "val", "test"]:
        path = os.path.join(DATA_DIR, f"{name}_labelled.parquet")
        if os.path.exists(path):
            print(f"loading {name} ...")
            parts.append(pd.read_parquet(path, columns=cols))
    if not parts:
        raise SystemExit(f"no labelled parquet files found in {DATA_DIR}")
    df = pd.concat(parts, ignore_index=True)
    return df.astype(bool)


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"13_figure2_combinations v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")

    df = load_flags()
    n_total = len(df)
    cs = df["cs_long_method"] | df["cs_high_cc"] | df["cs_long_params"]
    scs = df["ap_spaghetti"] | df["ap_high_fanout"]
    pops = {
        "Neither": (~cs & ~scs).sum(),
        "Code Smell only": (cs & ~scs).sum(),
        "SCS only": (~cs & scs).sum(),
        "Both": (cs & scs).sum(),
    }
    print(f"N = {n_total:,}")
    for k, v in pops.items():
        print(f"  {k}: {v:,} ({100 * v / n_total:.1f}%)")

    # ---- combinations among affected functions --------------------
    affected = df[cs | scs]
    n_aff = len(affected)
    key = np.zeros(n_aff, dtype=int)
    for i, (_, col) in enumerate(FLAGS):
        key |= affected[col].to_numpy().astype(int) << i
    counts = pd.Series(key).value_counts()
    rows = []
    for k, n in counts.items():
        members = [FLAGS[i][0] for i in range(5) if k >> i & 1]
        rows.append({"combination": "+".join(members),
                     "members": members, "n": int(n),
                     "pct_of_affected": 100 * n / n_aff,
                     "pct_of_corpus": 100 * n / n_total})
    comb = pd.DataFrame(rows).sort_values("n", ascending=False
                                          ).reset_index(drop=True)
    print(f"\nall {len(comb)} observed combinations "
          f"(affected functions: {n_aff:,}):")
    for _, r in comb.iterrows():
        print(f"  {r['combination']:22s} {r['n']:>9,}  "
              f"{r['pct_of_affected']:5.1f}% of affected  "
              f"{r['pct_of_corpus']:5.2f}% of corpus")
    config.OUT_DIR.mkdir(exist_ok=True)
    comb.drop(columns="members").to_csv(
        config.OUT_DIR / "figure2_combinations.csv", index=False)

    # ---- figure ----------------------------------------------------
    top = comb.head(TOP_K)
    other = comb.iloc[TOP_K:]["n"].sum()

    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.6],
                          height_ratios=[3.2, 1.5], hspace=0.06,
                          wspace=0.28)

    # Panel A: four populations
    axA = fig.add_subplot(gs[:, 0])
    labels = list(pops.keys())
    vals = [100 * pops[k] / n_total for k in labels]
    colors = ["#c7cdd6", "#4c6ef5", "#94a3ee", "#2b3a90"]
    y = np.arange(len(labels))[::-1]
    axA.barh(y, vals, color=colors, edgecolor="white")
    for yi, v in zip(y, vals):
        axA.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=9, fontname="Times New Roman")
    axA.set_yticks(y, labels, fontsize=9, fontname="Times New Roman")
    axA.set_xlim(0, 100)
    axA.set_xlabel("% of all functions", fontsize=9, fontname="Times New Roman")
    axA.set_title("A. Population structure", fontsize=10, loc="left", fontname="Times New Roman")
    axA.spines[["top", "right"]].set_visible(False)

    # Panel B: combination bars + membership matrix
    axB = fig.add_subplot(gs[0, 1])
    x = np.arange(len(top))
    axB.bar(x, top["pct_of_affected"], color="#4c6ef5",
            edgecolor="white")
    for xi, v in zip(x, top["pct_of_affected"]):
        axB.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, fontname="Times New Roman")
    if other:
        axB.text(0.99, 0.95,
                 f"remaining combinations: "
                 f"{100 * other / n_aff:.1f}% of affected",
                 transform=axB.transAxes, ha="right", va="top",
                 fontsize=8, color="#555555", fontname="Times New Roman")
    axB.set_xticks([])
    axB.set_ylabel("% of affected functions", fontsize=9, fontname="Times New Roman")
    axB.set_title("B. Sub-type combinations among affected functions",
                  fontsize=10, loc="left", fontname="Times New Roman")
    axB.spines[["top", "right"]].set_visible(False)

    axM = fig.add_subplot(gs[1, 1], sharex=axB)
    order = [f for f, _ in FLAGS]
    for xi, (_, r) in enumerate(top.iterrows()):
        ys = [order.index(m) for m in r["members"]]
        axM.scatter([xi] * 5, range(5), s=28, color="#d5d9e2",
                    zorder=1)
        axM.scatter([xi] * len(ys), ys, s=42, color="#2b3a90",
                    zorder=2)
        if len(ys) > 1:
            axM.plot([xi, xi], [min(ys), max(ys)], color="#2b3a90",
                     lw=1.4, zorder=1)
    axM.set_yticks(range(5), [FLAG_NAMES[f] for f in order],
                   fontsize=8, fontname="Times New Roman")
    axM.set_ylim(-0.6, 4.6)
    axM.invert_yaxis()
    axM.set_xticks([])
    for s in ["top", "right", "bottom", "left"]:
        axM.spines[s].set_visible(False)

    for ext in ["pdf", "png"]:
        out = config.OUT_DIR / f"figure2_cooccurrence_v2.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    print("Done. Paste the printed combination table back into the chat.")


if __name__ == "__main__":
    main()