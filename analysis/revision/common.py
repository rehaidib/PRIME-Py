"""Shared helpers for the revision analysis scripts (v3.1).

Verified rules (03 + 04 + residual-row forensics):
  - LPL: a parameter counts unless it is named exactly 'self' or 'cls'
    (ANY position, not just leading); Long Parameter List when the
    count is strictly greater than 5.
  - Excessive Branching: count(If + For + While + Try + With) > 12,
    read from outputs/spaghetti_rule.json (written by 04).
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

import config


def parse_param_tokens(val) -> list[str]:
    """function_params -> list of bare parameter names."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        raw = [str(v) for v in val]
    else:
        t = str(val).strip()
        if t in ("", "[]", "()", "nan", "None"):
            return []
        if t[0] in "[(":
            t = t[1:]
        if t and t[-1] in "])":
            t = t[:-1]
        raw = t.split(",")
    toks = []
    for r in raw:
        tok = r.strip().strip("'\"").strip()
        tok = tok.split(":")[0].split("=")[0].strip().lstrip("*")
        if tok:
            toks.append(tok)
    return toks


def nonself_param_count(series: pd.Series) -> np.ndarray:
    """Verified rule: count parameters not named self/cls (any position)."""
    tokens = series.map(parse_param_tokens)
    return tokens.map(
        lambda t: sum(1 for x in t if x not in ("self", "cls"))
    ).to_numpy()


# ------------------------------------------------------- cached loaders
def cached_nonself(split: str) -> np.ndarray:
    """Non-self parameter counts, cached per split (string parsing is
    the slow part; the cache makes 05/06/07 start instantly)."""
    cache = config.AST_CACHE_DIR / f"nonself_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["nonself"].to_numpy()
    s = pd.read_parquet(config.DATA_DIR / config.SPLIT_FILES[split],
                        columns=[config.COLUMNS["params_list"]]
                        )[config.COLUMNS["params_list"]]
    n = nonself_param_count(s)
    config.AST_CACHE_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"nonself": n}).to_parquet(cache, index=False)
    return n


def branch_counts(split: str) -> np.ndarray:
    """Branch counts under the empirically recovered rule."""
    rule = json.loads(config.SPAGHETTI_RULE_JSON.read_text())
    include = rule["include_node_types"]
    d = pd.read_parquet(config.AST_CACHE_DIR / f"{split}_ast_pertype.parquet")
    d = d.sort_values("row_index").reset_index(drop=True)
    counts = sum(d[n].to_numpy() for n in include)
    return counts


def max_nesting(split: str) -> np.ndarray:
    d = pd.read_parquet(config.AST_CACHE_DIR / f"{split}_ast_pertype.parquet")
    d = d.sort_values("row_index").reset_index(drop=True)
    return d["max_nesting"].to_numpy()


def load_masks(split: str):
    """Return (project array, metrics dict, canonical label-mask dict)
    for one split, positionally aligned with the split parquet."""
    path = config.DATA_DIR / config.SPLIT_FILES[split]
    df = pd.read_parquet(path, columns=[
        config.COLUMNS["project"], config.COLUMNS["nloc"],
        config.COLUMNS["cc"], config.COLUMNS["calls"]])
    metrics = {
        "nloc":     df[config.COLUMNS["nloc"]].to_numpy(),
        "cc":       df[config.COLUMNS["cc"]].to_numpy(),
        "calls":    df[config.COLUMNS["calls"]].to_numpy(),
        "params":   cached_nonself(split),
        "branches": branch_counts(split),
    }
    masks = {}
    for subtype, (metric, t) in config.THRESHOLDS.items():
        masks[subtype] = metrics[metric] > t
    masks["any_code_smell"] = np.logical_or.reduce(
        [masks[s] for s in config.CODE_SMELL_SUBTYPES])
    masks["any_scs"] = np.logical_or.reduce(
        [masks[s] for s in config.SCS_SUBTYPES])
    masks["both"] = masks["any_code_smell"] & masks["any_scs"]
    masks["neither"] = ~masks["any_code_smell"] & ~masks["any_scs"]
    return df[config.COLUMNS["project"]].to_numpy(), metrics, masks