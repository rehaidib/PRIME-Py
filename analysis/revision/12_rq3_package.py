"""12 (A7 + A9): complete RQ3 recomputation from the annotation workbooks.

Inputs: the pre-adjudication workbook (192 structural functions, both
annotators' labels, first-author adjudication) and the third-party
adjudication workbook (86 disagreement rows, blinded independent
expert's Final Label).

Produces, per category (Code Smell / Anti-Pattern):
  1. Pre-adjudication Cohen's kappa with bootstrap 95% CIs and the
     A1 x A2 confusion matrix (all disagreement is A1+/A2-).
  2. Panel B model-vs-gold under two schemes:
       Scheme A: first-author adjudication (as published)
       Scheme B: blinded independent third rater (2-of-3 majority gold)
     with bootstrap 95% CIs on kappa.
  3. First-author vs expert adjudicator comparison on the 86 disputes.
  4. Anti-Pattern scope analysis: sample composition (as-sampled vs
     paper-scope spaghetti|fanout positives; God-Class-only count) and
     Panel B recomputed with paper-scope model labels under both golds.
  5. Subtype breakdown of model-positive rows by driving Detail flag.
  6. Prevalence-adjusted PPV at verified corpus prevalence.

Feeds: E4, E7, R2.7, R2.8, R3.16, R3.27, R3.29.

Usage:
    conda activate prime
    python 12_rq3_package.py pre-adjudication.xlsx 3rd_party_adjudication.xlsx
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

import config

SCRIPT_VERSION = 3

# verified whole-corpus category prevalences (01_table2_rederive.py)
PREVALENCE = {"Code Smell": 0.2484, "Anti-Pattern": 0.0760}
DETAILS = ["Detail: Long Method", "Detail: High CC", "Detail: Long Params",
           "Detail: Spaghetti", "Detail: High Fan-Out", "Detail: God Class"]


def kappa(x, y) -> float:
    x, y = np.asarray(x, int), np.asarray(y, int)
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe))


def boot_kappa_ci(x, y, rng) -> tuple[float, float]:
    x, y = np.asarray(x, int), np.asarray(y, int)
    ks = np.empty(config.N_BOOTSTRAP)
    for i in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, len(x), len(x))
        ks[i] = kappa(x[idx], y[idx])
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def cells(ym, yg) -> tuple[int, int, int, int]:
    return (int(((ym == 1) & (yg == 1)).sum()),
            int(((ym == 1) & (yg == 0)).sum()),
            int(((ym == 0) & (yg == 1)).sum()),
            int(((ym == 0) & (yg == 0)).sum()))


def prf(tp, fp, fn, tn) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main() -> None:
    cfg_v = getattr(config, "CONFIG_VERSION", 0)
    print(f"12_rq3_package v{SCRIPT_VERSION} / config v{cfg_v}")
    if cfg_v != SCRIPT_VERSION:
        raise SystemExit("Version mismatch: config.py v3 required.")
    pre_path = sys.argv[1] if len(sys.argv) > 1 else "pre-adjudication.xlsx"
    adj_path = sys.argv[2] if len(sys.argv) > 2 else "3rd_party_adjudication.xlsx"
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    config.OUT_DIR.mkdir(exist_ok=True)

    pre = pd.read_excel(pre_path)
    pre.columns = [str(c).strip() for c in pre.columns]
    adj = pd.read_excel(adj_path, header=1)
    adj.columns = [str(c).strip() for c in adj.columns]

    a1 = pd.to_numeric(pre["Annotator (1) Label"], errors="coerce")
    a2 = pd.to_numeric(pre["Annotator (2) Label"], errors="coerce")
    fa = pd.to_numeric(pre["irst-author label"], errors="coerce")  # sic
    model = pre["Model Label"].astype(str).str.strip().str.lower(
        ).eq("positive").astype(int)
    dis = (a1 != a2)
    expert = pre["ID"].astype(int).map(
        adj.set_index(adj["ID"].astype(int))["Final Label"])
    gold = {"A_first_author": np.where(dis, fa, a1),
            "B_independent": np.where(dis, expert, a1)}
    cats = ["Code Smell", "Anti-Pattern"]

    # ---- 1. pre-adjudication agreement -------------------------------
    rows = []
    print("\n1. pre-adjudication agreement:")
    for cat in cats:
        m = (pre["PDS Type"] == cat).to_numpy()
        k = kappa(a1[m], a2[m])
        lo, hi = boot_kappa_ci(a1[m], a2[m], rng)
        n_dis = int((a1[m] != a2[m]).sum())
        a1p, a2p = int(a1[m].sum()), int(a2[m].sum())
        print(f"  [{cat}] n={int(m.sum())} disagreements={n_dis} "
              f"(all A1+/A2-) A1+={a1p} A2+={a2p} "
              f"kappa={k:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append({"category": cat, "n": int(m.sum()),
                     "disagreements": n_dis, "a1_pos": a1p, "a2_pos": a2p,
                     "kappa": round(k, 3), "ci_low": round(lo, 3),
                     "ci_high": round(hi, 3)})
    pd.DataFrame(rows).to_csv(config.OUT_DIR / "rq3_preadjudication.csv",
                              index=False)

    # ---- 2. Panel B under both schemes -------------------------------
    rows = []
    print("\n2. Panel B (model vs gold), both schemes:")
    for scheme, g in gold.items():
        for cat in cats:
            m = (pre["PDS Type"] == cat).to_numpy()
            yg = pd.Series(g)[m].astype(int).to_numpy()
            ym = model[m].to_numpy()
            tp, fp, fn, tn = cells(ym, yg)
            prec, rec, f1 = prf(tp, fp, fn, tn)
            k = kappa(ym, yg)
            lo, hi = boot_kappa_ci(ym, yg, rng)
            print(f"  [{scheme} | {cat}] TP={tp} FP={fp} FN={fn} TN={tn} "
                  f"Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} "
                  f"kappa={k:.3f} [{lo:.3f}, {hi:.3f}]")
            rows.append({"scheme": scheme, "category": cat, "tp": tp,
                         "fp": fp, "fn": fn, "tn": tn,
                         "precision": round(prec, 3),
                         "recall": round(rec, 3), "f1": round(f1, 3),
                         "kappa": round(k, 3), "ci_low": round(lo, 3),
                         "ci_high": round(hi, 3)})
    pd.DataFrame(rows).to_csv(config.OUT_DIR / "rq3_panelB.csv", index=False)

    # ---- 3. adjudicator comparison -----------------------------------
    fa_d = fa[dis].astype(int).to_numpy()
    ex_d = pd.Series(expert)[dis].astype(int).to_numpy()
    cat_d = pre.loc[dis, "PDS Type"].to_numpy()
    rows = []
    print("\n3. adjudicator comparison on the 86 disputes:")
    for cat in cats:
        m = cat_d == cat
        agree = int((fa_d[m] == ex_d[m]).sum())
        rows.append({"category": cat, "n": int(m.sum()),
                     "first_author_pos": int(fa_d[m].sum()),
                     "expert_pos": int(ex_d[m].sum()), "agree": agree})
        print(f"  [{cat}] n={int(m.sum())} FA+={int(fa_d[m].sum())} "
              f"expert+={int(ex_d[m].sum())} agree={agree}")
    k = kappa(fa_d, ex_d)
    print(f"  overall: agree={int((fa_d == ex_d).sum())}/86 "
          f"({100 * (fa_d == ex_d).mean():.0f}%), kappa={k:.3f}")
    df3 = pd.DataFrame(rows)
    df3["overall_kappa"] = round(k, 3)
    df3.to_csv(config.OUT_DIR / "rq3_adjudicators.csv", index=False)

    # ---- 4. Anti-Pattern scope analysis ------------------------------
    ap = (pre["PDS Type"] == "Anti-Pattern").to_numpy()
    scope = (pd.to_numeric(pre["Detail: Spaghetti"], errors="coerce"
                           ).fillna(0).astype(int)
             | pd.to_numeric(pre["Detail: High Fan-Out"], errors="coerce"
                             ).fillna(0).astype(int)).to_numpy()
    god_only = int((model[ap] == 1).sum()
                   - int((scope[ap] & (model[ap] == 1)).sum()))
    print(f"\n4. AP scope: as-sampled positives={int(model[ap].sum())}, "
          f"paper-scope positives={int(scope[ap].sum())}, "
          f"god-class-only={god_only}")
    rows = []
    for scheme, g in gold.items():
        yg = pd.Series(g)[ap].astype(int).to_numpy()
        ym = scope[ap]
        tp, fp, fn, tn = cells(ym, yg)
        prec, rec, f1 = prf(tp, fp, fn, tn)
        k = kappa(ym, yg)
        print(f"  paper-scope model vs {scheme}: TP={tp} FP={fp} FN={fn} "
              f"TN={tn} Prec={prec:.3f} Rec={rec:.3f} kappa={k:.3f}")
        rows.append({"scheme": scheme, "tp": tp, "fp": fp, "fn": fn,
                     "tn": tn, "precision": round(prec, 3),
                     "recall": round(rec, 3), "kappa": round(k, 3)})
    pd.DataFrame(rows).to_csv(config.OUT_DIR / "rq3_ap_scope.csv",
                              index=False)

    # ---- 5. subtype breakdown of model-positive rows -----------------
    rows = []
    print("\n5. subtype breakdown (model-positive rows; flags overlap):")
    for cat in cats:
        m = ((pre["PDS Type"] == cat) & (model == 1)).to_numpy()
        for det in DETAILS:
            f = pd.to_numeric(pre[det], errors="coerce").fillna(0
                                                                ).astype(int)
            mm = m & (f == 1).to_numpy()
            if mm.sum() == 0:
                continue
            rows.append({
                "category": cat, "detail": det.replace("Detail: ", ""),
                "n": int(mm.sum()), "a1_pos": int(a1[mm].sum()),
                "a2_pos": int(a2[mm].sum()),
                "goldA_pos": int(pd.Series(gold["A_first_author"]
                                           )[mm].sum()),
                "goldB_pos": int(pd.Series(gold["B_independent"])[mm].sum()),
            })
            r = rows[-1]
            print(f"  [{cat}] {r['detail']:14s} n={r['n']:>2} "
                  f"A1+={r['a1_pos']:>2} A2+={r['a2_pos']:>2} "
                  f"goldA+={r['goldA_pos']:>2} goldB+={r['goldB_pos']:>2}")
    pd.DataFrame(rows).to_csv(config.OUT_DIR / "rq3_subtype_breakdown.csv",
                              index=False)

    # ---- 6. prevalence-adjusted PPV ----------------------------------
    rows = []
    print("\n6. prevalence-adjusted PPV at corpus prevalence:")
    for scheme, g in gold.items():
        for cat in cats:
            m = (pre["PDS Type"] == cat).to_numpy()
            yg = pd.Series(g)[m].astype(int).to_numpy()
            ym = model[m].to_numpy()
            tp, fp, fn, tn = cells(ym, yg)
            sens = tp / (tp + fn) if tp + fn else 0.0
            spec = tn / (tn + fp) if tn + fp else 0.0
            pi = PREVALENCE[cat]
            denom = sens * pi + (1 - spec) * (1 - pi)
            ppv = sens * pi / denom if denom else 0.0
            print(f"  [{scheme} | {cat}] sens={sens:.3f} spec={spec:.3f} "
                  f"PPV@{pi:.3f}={ppv:.3f}")
            rows.append({"scheme": scheme, "category": cat,
                         "sensitivity": round(sens, 3),
                         "specificity": round(spec, 3),
                         "corpus_prevalence": pi, "ppv": round(ppv, 3)})
    pd.DataFrame(rows).to_csv(config.OUT_DIR / "rq3_ppv.csv", index=False)

    print("\nDone. Six CSVs written to outputs/.")


if __name__ == "__main__":
    main()