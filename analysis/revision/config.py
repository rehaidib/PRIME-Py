"""Shared configuration for PeerJ CS-2026:05:140356 revision analyses.

v3 changes (after 03_diagnose_mismatches.py findings):
  - LPL is computed from the function_params list minus self/cls (H2),
    NOT from num_parameter (Lizard raw count, includes self/cls and
    disagrees with the AST list on ~6% of rows).
  - The spaghetti branch-counting rule is discovered empirically by
    04_ast_pertype.py and stored in outputs/spaghetti_rule.json, which
    downstream scripts read.

!! Re-apply your actual split file names in SPLIT_FILES if they differ.
"""
from pathlib import Path

CONFIG_VERSION = 3   # scripts check this to catch stale-file mixups

# ---------------------------------------------------------------- paths
DATA_DIR = Path("/Users/reemehaidib/PhD_Dataset/PRIME_Output/labelled")

# TODO: re-apply your actual file names here if you changed them.
SPLIT_FILES = {
    "train": "train_labelled.parquet",
    "validation": "val_labelled.parquet",
    "test": "test_labelled.parquet",
}

OUT_DIR = Path(__file__).resolve().parent / "outputs"
AST_CACHE_DIR = Path(__file__).resolve().parent / "ast_cache"
SPAGHETTI_RULE_JSON = OUT_DIR / "spaghetti_rule.json"

# ------------------------------------------------------- column mapping
COLUMNS = {
    "project":     "project_name",
    "nloc":        "nloc",
    "cc":          "cyclomatic_complexity",
    "params_raw":  "num_parameter",          # Lizard raw count (incl. self)
    "params_list": "function_params",        # AST list -> H2 rule source
    "calls":       "outgoing_function_count",
    "body":        "function_body",
    "class_name":  "class_name",
}

LABELS = {
    "long_method":         "cs_long_method",
    "high_cc":             "cs_high_cc",
    "long_parameter_list": "cs_long_params",
    "excessive_branching": "ap_spaghetti",      # released under old name
    "high_fan_out":        "ap_high_fanout",
}

AGGREGATE_LABELS = {
    "code_smell":   "code_smell_label",
    "anti_pattern": "anti_pattern_label",
    "god_class":    "ap_god_class",
}

# ------------------------------------------------------------ thresholds
# 'params' is the H2-derived non-self count (common.nonself_param_count);
# 'branches' comes from the per-type AST cache + spaghetti_rule.json.
THRESHOLDS = {
    "long_method":         ("nloc",     14),
    "high_cc":             ("cc",       10),
    "long_parameter_list": ("params",    5),
    "excessive_branching": ("branches", 12),
    "high_fan_out":        ("calls",    15),
}

CODE_SMELL_SUBTYPES = ["long_method", "high_cc", "long_parameter_list"]
SCS_SUBTYPES        = ["excessive_branching", "high_fan_out"]

# -------------------------------------------------- A1 sensitivity grids
SENSITIVITY_GRIDS = {
    "long_method_percentiles": [70, 75, 80, 90],   # of the TRAINING split
    "long_method_fixed_nloc":  [20, 30, 50],
    "long_parameter_list":     [3, 4, 5, 6, 7],    # rule: params > t
    "high_cc":                 [8, 9, 10, 11, 12, 13, 14, 15],
    "excessive_branching":     [8, 9, 10, 11, 12, 13, 14, 15, 16],
    "high_fan_out":            [10, 12, 14, 15, 16, 18, 20],
}

# --------------------------------------------------------- A3 bootstrap
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

# ------------------------------------------------- AST per-type settings
AST_BATCH_SIZE = 50_000
# Node kinds tallied individually by 04_ast_pertype.py; the variant
# search assembles candidate branch-counting rules from subsets.
PERTYPE_NODE_NAMES = (
    "If", "For", "AsyncFor", "While", "Try", "TryStar",
    "With", "AsyncWith", "ExceptHandler", "Match", "match_case",
)
NESTING_NODE_NAMES = (
    "If", "For", "AsyncFor", "While", "Try", "TryStar",
    "With", "AsyncWith",
)

# ---------------------------------------------- published Table 2 values
PUBLISHED_TABLE2 = {
    "train": {
        "total": 1_554_896, "long_method": 361_804, "high_cc": 47_220,
        "long_parameter_list": 41_094, "any_code_smell": 371_078,
        "excessive_branching": 17_565, "high_fan_out": 117_792,
        "any_scs": 120_016, "both": 109_004, "neither": 1_172_806,
    },
    "validation": {
        "total": 165_226, "long_method": 42_542, "high_cc": 5_824,
        "long_parameter_list": 3_094, "any_code_smell": 43_184,
        "excessive_branching": 2_392, "high_fan_out": 12_908,
        "any_scs": 13_112, "both": 11_992, "neither": 120_922,
    },
    "test": {
        "total": 277_413, "long_method": 80_169, "high_cc": 6_479,
        "long_parameter_list": 10_331, "any_code_smell": 81_989,
        "excessive_branching": 2_376, "high_fan_out": 18_342,
        "any_scs": 18_676, "both": 17_352, "neither": 1_487_828,  # erratum
    },
    "corpus": {
        "total": 1_997_535, "long_method": 484_515, "high_cc": 59_523,
        "long_parameter_list": 54_519, "any_code_smell": 496_251,
        "excessive_branching": 22_333, "high_fan_out": 149_042,
        "any_scs": 151_804, "both": 138_348, "neither": 1_487_828,
    },
}
EXPECTED_TEST_NEITHER_CORRECTED = 194_100