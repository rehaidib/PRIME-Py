"""08: regenerate Figure 2 as a single panel (locked decision, R3.6).

Reads outputs/figure_inputs.json (written by 01 v3) and renders one
100% horizontal stacked bar of the four functional populations over
the whole corpus, in colorblind-safe colors, saved as PDF + 300-dpi
PNG.  Prints the caption text to use in the manuscript.

Usage:
    conda activate prime
    python 08_figure2.py
"""
from __future__ import annotations
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SCRIPT_VERSION = 3

SEGMENTS = [
    ("neither_pct",         "Structurally clean", "#c6dbef"),
    ("code_smell_only_pct", "Code Smell only",    "#6baed6"),
    ("both_pct",            "Both categories",    "#2171b5"),
    ("scs_only_pct",        "SCS only",           "#08306b"),
]


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"08_figure2 v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    fig_inputs = json.loads(
        (config.OUT_DIR / "figure_inputs.json").read_text())
    data = fig_inputs["figure2_whole_corpus_single_panel"]

    fig, ax = plt.subplots(figsize=(8.0, 2.2))
    left = 0.0
    for key, label, color in SEGMENTS:
        v = data[key]
        ax.barh(0, v, left=left, color=color, edgecolor="white",
                height=0.55, label=f"{label} ({v}%)")
        if v >= 4:
            ax.text(left + v / 2, 0, f"{v}%", va="center", ha="center",
                    fontsize=9, color="black" if v > 30 else "white")
        left += v

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Percentage of functions (%)", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55),
              ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()

    config.OUT_DIR.mkdir(exist_ok=True)
    pdf = config.OUT_DIR / "figure2_cooccurrence.pdf"
    png = config.OUT_DIR / "figure2_cooccurrence.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"wrote {pdf}\nwrote {png}")

    print("\nsuggested caption:")
    print("  Figure 2. Co-occurrence of Code Smells and Structural "
          "Complexity Symptoms across all 1,997,535 functions in the "
          "PRIME-Py corpus. SCS = Structural Complexity Symptoms.")
    print("\nDone.")


if __name__ == "__main__":
    main()