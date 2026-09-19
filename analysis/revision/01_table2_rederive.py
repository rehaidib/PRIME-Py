"""A10 (v3): re-derive Table 2 using the verified LPL rule.

Changes vs v2:
  - Long Parameter List computed via the H2 rule confirmed by 03:
    (len(function_params) - self/cls) > 5.
  - Any residual LPL mismatches vs cs_long_params (expected: ~13 rows
    corpus-wide) are dumped in full so the cause can be inspected.

Expected outcome: every published Table 2 cell reproduces exactly
(within the residual rows), and Test/Neither lands at 194,100.

Usage:
    conda activate prime
    python 01_table2_rederive.py
"""
from __future__ import annotations
import json

import pandas as pd
import pyarrow.parquet as pq

import common
import config

SCRIPT_VERSION = 3


def load_split(split: str) -> pd.DataFrame:
    path = config.DATA_DIR / config.SPLIT_FILES[split]
    present = set(pq.read_schema(path).names)
    wanted = {config.COLUMNS["project"], config.COLUMNS["nloc"],
              config.COLUMNS["cc"], config.COLUMNS["calls"],
              config.COLUMNS["params_raw"], config.COLUMNS["params_list"],
              config.COLUMNS["class_name"]}
    wanted |= {c for c in config.LABELS.values() if c in present}
    wanted |= {c for c in config.AGGREGATE_LABELS.values() if c in present}
    missing = wanted - present
    if missing:
        raise SystemExit(f"[{split}] columns not found in parquet: {missing}")
    return pd.read_parquet(path, columns=sorted(wanted))


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["long_method"] = df[config.COLUMNS["nloc"]] > 14
    out["high_cc"] = df[config.COLUMNS["cc"]] > 10
    nonself = common.nonself_param_count(df[config.COLUMNS["params_list"]])
    out["long_parameter_list"] = nonself > 5
    out["excessive_branching"] = df[config.LABELS["excessive_branching"]
                                    ].astype(bool)     # verified by 04
    out["high_fan_out"] = df[config.COLUMNS["calls"]] > 15
    out["any_code_smell"] = out[config.CODE_SMELL_SUBTYPES].any(axis=1)
    out["any_scs"] = out[config.SCS_SUBTYPES].any(axis=1)
    out["both"] = out["any_code_smell"] & out["any_scs"]
    out["neither"] = ~out["any_code_smell"] & ~out["any_scs"]
    out["_nonself"] = nonself
    return out


def crosscheck_stored(df: pd.DataFrame, lab: pd.DataFrame, split: str) -> None:
    print(f"  -- stored-label cross-checks [{split}] --")
    checks = [("long_method", "long_method"), ("high_cc", "high_cc"),
              ("long_parameter_list", "long_parameter_list"),
              ("high_fan_out", "high_fan_out")]
    for subtype, key in checks:
        col = config.LABELS[subtype]
        stored = df[col].astype(bool)
        mism = stored != lab[key]
        n = int(mism.sum())
        flag = "OK" if n == 0 else "residual rows dumped below"
        print(f"    {subtype:22s}: {n:,} mismatches  {flag}")
        if n and subtype == "long_parameter_list":
            dump = df.loc[mism, [config.COLUMNS["params_list"],
                                 config.COLUMNS["params_raw"],
                                 config.COLUMNS["class_name"], col]].head(15)
            dump = dump.assign(derived_nonself=lab.loc[mism, "_nonself"].head(15))
            for _, r in dump.iterrows():
                print(f"      raw={str(r[config.COLUMNS['params_list']])[:70]!r} "
                      f"num_parameter={r[config.COLUMNS['params_raw']]} "
                      f"derived_nonself={r['derived_nonself']} "
                      f"class={str(r[config.COLUMNS['class_name']])[:20]!r} "
                      f"stored={bool(r[col])}")

    agg = config.AGGREGATE_LABELS
    stored = df[agg["code_smell"]].astype(bool)
    n = int((stored != lab["any_code_smell"]).sum())
    print(f"    code_smell_label == our union: {n:,} mismatches "
          f"{'OK' if n == 0 else '(residual LPL rows)'}")


ROWS = ["long_method", "high_cc", "long_parameter_list", "any_code_smell",
        "excessive_branching", "high_fan_out", "any_scs", "both", "neither"]


def summarize(lab: pd.DataFrame) -> dict:
    d = {"total": len(lab)}
    for r in ROWS:
        d[r] = int(lab[r].sum())
    return d


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"01_table2_rederive v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit(
            "Version mismatch: this script requires config.py v3. "
            "Replace stale files in analysis/revision/ with the latest set "
            "(and re-apply your SPLIT_FILES names).")
    config.OUT_DIR.mkdir(exist_ok=True)
    summaries: dict[str, dict] = {}
    corpus_parts = []

    for split in config.SPLIT_FILES:
        print(f"loading {split} ...")
        df = load_split(split)
        lab = compute_labels(df)
        crosscheck_stored(df, lab, split)
        summaries[split] = summarize(lab)
        corpus_parts.append(lab.drop(columns=["_nonself"]))

    corpus = pd.concat(corpus_parts, ignore_index=True)
    summaries["corpus"] = summarize(corpus)

    print("\n=== recomputed vs published (diff != 0 needs attention) ===")
    for split, summ in summaries.items():
        pub = config.PUBLISHED_TABLE2[split]
        for key, val in summ.items():
            pv = pub.get(key)
            if pv is None:
                continue
            diff = val - pv
            note = ""
            if split == "test" and key == "neither":
                ok = val == config.EXPECTED_TEST_NEITHER_CORRECTED
                note = ("  <-- known erratum; corrected value "
                        f"{'CONFIRMED' if ok else 'close but not exact'} "
                        f"(expected {config.EXPECTED_TEST_NEITHER_CORRECTED:,})")
            elif diff != 0:
                note = "  <-- differs from published"
            print(f"[{split:10s}] {key:22s} recomputed={val:>11,} "
                  f"published={pv:>11,} diff={diff:+,}{note}")

    rows = []
    for r in ROWS:
        row = {"subtype": r}
        for split in ["train", "validation", "test", "corpus"]:
            s = summaries[split]
            row[f"{split}_n"] = s[r]
            row[f"{split}_pct"] = round(100 * s[r] / s["total"], 1)
        rows.append(row)
    out_csv = config.OUT_DIR / "table2_rederived.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    def four_pop(s):
        t = s["total"]
        return {
            "neither_pct":         round(100 * s["neither"] / t, 1),
            "code_smell_only_pct": round(100 * (s["any_code_smell"] - s["both"]) / t, 1),
            "scs_only_pct":        round(100 * (s["any_scs"] - s["both"]) / t, 1),
            "both_pct":            round(100 * s["both"] / t, 1),
        }

    fig_inputs = {
        "figure1_whole_corpus": {
            "any_code_smell_pct": round(100 * summaries["corpus"]["any_code_smell"]
                                        / summaries["corpus"]["total"], 1),
            "any_scs_pct":        round(100 * summaries["corpus"]["any_scs"]
                                        / summaries["corpus"]["total"], 1),
            "cooccurrence_pct":   round(100 * summaries["corpus"]["both"]
                                        / summaries["corpus"]["total"], 1),
            "seed_note": "seed = 42 (verify against split script before regenerating figure)",
        },
        "figure2_whole_corpus_single_panel": four_pop(summaries["corpus"]),
        "note": "Locked decisions: Fig 1 whole-corpus; Fig 2 single panel.",
    }
    (config.OUT_DIR / "figure_inputs.json").write_text(
        json.dumps(fig_inputs, indent=2))
    print(f"wrote {config.OUT_DIR / 'figure_inputs.json'}")
    print("\nDone. Paste the cross-check and diff sections back into the chat.")


if __name__ == "__main__":
    main()