# ============================================================
# generate_pds_labels.py
# PRIME Dataset — PDS Label Generation
#
# PURPOSE:
#   Generates binary Poor Design Symptom (PDS) labels for
#   every function in the dataset using threshold-based rules
#   grounded in published empirical research.
#
#   Four label columns are produced per function:
#
#   code_smell_label     : Long Method, High Complexity,
#                          Long Parameter List
#   anti_pattern_label   : Spaghetti Code, High Fan-Out
#   poor_naming_label    : Invalid name, ambiguous name,
#                          non-snake_case
#   poor_doc_label       : Missing or trivial documentation
#
#   Additionally, sub-type columns identify the specific
#   smell within each category for granular analysis.
#
# THRESHOLD CITATIONS:
#   Long Method         : Chen et al. (2018) IST Q2
#                         Statistics-based: 75th percentile
#   High Complexity     : McCabe (1976) IEEE TSE Q1
#                         CC > 10
#   Long Parameter List : Chen et al. (2018) + Fowler (2018)
#                         > 5 parameters (excl. self/cls)
#   Spaghetti Code      : McCabe (1976) + Pylint R0912
#                         Branches > 12
#   High Fan-Out        : Palomba et al. (2018) EMSE Q1
#                         Outgoing calls > 15
#   Poor Naming         : Arnaoudova et al. (2016) EMSE Q1
#                         PEP 8 violations + known bad patterns
#   Poor Documentation  : Tamrakar et al. (2021) IEEE SANER
#                         Missing/trivial docstring (PEP 257)
#
# HOW TO RUN:
#   conda activate prime
#   python generate_pds_labels.py
# ============================================================

import os
import sys
import re
import ast
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================
# DUAL OUTPUT
# ============================================================

class DualOutput:
    def __init__(self, log_path: str):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ============================================================
# CONFIGURATION
# ============================================================

# Input: the splits with lexical representations
SPLITS_DIR  = "/Users/reemehaidib/PhD_Dataset/PRIME_Output/splits"
OUTPUT_DIR  = "/Users/reemehaidib/PhD_Dataset/PRIME_Output/labelled"
LOG_DIR     = "/Users/reemehaidib/PhD_Dataset/PRIME_Logs"

# -------------------------------------------------------
# THRESHOLDS — each grounded in a published citation
# -------------------------------------------------------

# Code Smell thresholds
# Chen et al. (2018): statistics-based strategy uses 75th
# percentile of the corpus. We compute this at runtime
# from the training set.
LONG_METHOD_PERCENTILE    = 75      # Chen et al. (2018) IST

# McCabe (1976): CC > 10 is the widely accepted upper limit
CC_THRESHOLD              = 10      # McCabe (1976) IEEE TSE

# Chen et al. (2018) + Fowler (2018): > 5 parameters
# We subtract 1 if function has 'self' or 'cls' parameter
LONG_PARAM_THRESHOLD      = 5       # Chen et al. (2018) IST

# Anti-Pattern thresholds
# Pylint R0912 default: too-many-branches > 12
# Grounded in McCabe (1976) complexity analysis
MAX_BRANCHES              = 12      # McCabe (1976) / Pylint R0912

# Palomba et al. (2018): high fan-out as Feature Envy proxy
# 75th percentile of outgoing_function_count in our corpus
# We use a fixed threshold of 15 based on the literature
HIGH_FANOUT_THRESHOLD     = 15      # Palomba et al. (2018) EMSE

# God Class thresholds (class-level aggregation)
# Pylint R0904 default: > 20 public methods
GOD_CLASS_METHOD_THRESHOLD = 30     # Pylint R0904

# Poor Documentation thresholds
# PEP 257: all public functions should have docstrings
# Tamrakar et al. (2021): Lazy smell = missing/trivial doc
MIN_DOCSTRING_WORDS       = 10      # Tamrakar et al. (2021)

# Poor Naming — patterns grounded in Arnaoudova et al. (2016)
# and PEP 8 (Python official style guide)
MIN_MEANINGFUL_NAME_LEN   = 3       # PEP 8 / Arnaoudova (2016)

# Known bad/ambiguous identifiers from literature
# Arnaoudova et al. (2016) identifies these as linguistic
# antipatterns that increase change-proneness
BAD_NAME_PATTERNS = {
    # Meaningless single-purpose placeholder names
    "foo", "bar", "baz", "qux",
    # Generic data names with no semantic meaning
    "data", "result", "tmp", "temp", "obj", "val", "value",
    # Common vague names
    "do_stuff", "do_thing", "process", "handle",
    # Single-letter names (except accepted loop variables)
    # We handle single letters separately below
}

# Single-letter names that are ACCEPTABLE in Python
# (standard loop variables, mathematical convention)
ACCEPTABLE_SINGLE_LETTERS = {
    "i", "j", "k",   # loop indices
    "x", "y", "z",   # coordinates / math
    "n", "m",         # counts / dimensions
    "f", "g", "h",   # function variables
    "e",              # exception variable
    "t",              # time variable
}

# ============================================================
# FEATURE DERIVATION HELPERS
#
# These functions extract features from existing columns
# that are needed for labelling but were not pre-computed
# in the master dataset.
# ============================================================

def parse_body_line_types(raw) -> dict:
    """
    Safely parse function_body_line_type JSON string.
    Returns empty dict on any failure.
    Input : '{"If": 2, "Assign": 3}'
    Output: {"If": 2, "Assign": 3}
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        try:
            return eval(raw)
        except Exception:
            return {}


def count_branches(body_line_type_raw) -> int:
    """
    Count branch-inducing statements in a function.
    Used for Spaghetti Code detection.

    Counts: If, For, While, Try, With, ExceptHandler
    Grounded in: Pylint R0912 definition of 'branch'
    """
    BRANCH_TYPES = {"If", "For", "While", "Try",
                    "With", "ExceptHandler"}
    counts = parse_body_line_types(body_line_type_raw)
    return sum(v for k, v in counts.items()
               if k in BRANCH_TYPES)


def has_docstring(func_body: str) -> bool:
    """
    Check if a Python function body begins with a docstring.

    A docstring is the first statement of a function if it
    is a string literal (triple-quoted or single-quoted).

    PEP 257: A docstring is a string literal that occurs
    as the first statement in a module, function, class,
    or method body.
    """
    if not isinstance(func_body, str) or not func_body.strip():
        return False
    try:
        tree = ast.parse(func_body)
        if not tree.body:
            return False
        # A function body parsed at module level will have
        # the FunctionDef as the first node
        func_nodes = [n for n in tree.body
                      if isinstance(n, (ast.FunctionDef,
                                        ast.AsyncFunctionDef))]
        if func_nodes:
            body = func_nodes[0].body
        else:
            body = tree.body

        if not body:
            return False

        first = body[0]
        return (
            isinstance(first, ast.Expr) and
            isinstance(first.value, ast.Constant) and
            isinstance(first.value.value, str)
        )
    except SyntaxError:
        # Fall back to string search for unparseable snippets
        stripped = func_body.strip()
        first_line_end = stripped.find('\n')
        first_part = (stripped if first_line_end == -1
                      else stripped[:first_line_end * 3])
        return ('"""' in first_part or
                "'''" in first_part)
    except Exception:
        return False


def get_docstring_text(func_body: str) -> str:
    """
    Extract the docstring text from a function body.
    Returns empty string if no docstring found.
    """
    if not isinstance(func_body, str):
        return ""
    try:
        tree = ast.parse(func_body)
        func_nodes = [n for n in tree.body
                      if isinstance(n, (ast.FunctionDef,
                                        ast.AsyncFunctionDef))]
        body = func_nodes[0].body if func_nodes else tree.body
        if not body:
            return ""
        first = body[0]
        if (isinstance(first, ast.Expr) and
                isinstance(first.value, ast.Constant) and
                isinstance(first.value.value, str)):
            return first.value.value.strip()
    except Exception:
        pass
    return ""


def count_docstring_words(func_body: str) -> int:
    """Count words in a function's docstring."""
    text = get_docstring_text(func_body)
    if not text:
        return 0
    return len(text.split())


def is_snake_case(name: str) -> bool:
    """
    Check if a name follows Python's snake_case convention.
    PEP 8: function names should be lowercase with underscores.

    Allows:
      - Leading underscore (private: _helper)
      - Double leading underscore (dunder: __init__)
      - All lowercase with underscores
    Rejects:
      - camelCase (hasUpper followed by lower)
      - PascalCase (starts with upper)
      - mixed case
    """
    if not isinstance(name, str) or not name:
        return False

    # Strip leading underscores for dunder/private methods
    stripped = name.lstrip("_")
    if not stripped:
        return True  # pure underscore names are unusual but valid

    # Reject if starts with uppercase (PascalCase)
    if stripped[0].isupper():
        return False

    # Reject camelCase: lowercase letter followed by uppercase
    if re.search(r'[a-z][A-Z]', stripped):
        return False

    return True


def is_poor_name(func_name: str,
                 bad_names: set,
                 acceptable_singles: set,
                 min_len: int) -> bool:
    """
    Determine if a function name is a poor naming convention.

    Detection rules (Arnaoudova et al. 2016 / PEP 8):
    1. Single-character name not in acceptable set
    2. Name shorter than min_len AND not a dunder method
    3. Name appears in the known bad patterns list
    4. Name does not follow snake_case convention
    """
    if not isinstance(func_name, str) or not func_name:
        return True

    # Dunder methods (__init__, __str__, etc.) are exempt
    # They follow a well-defined convention
    if func_name.startswith("__") and func_name.endswith("__"):
        return False

    clean = func_name.lstrip("_").lower()

    # Rule 1: Single character
    if len(clean) == 1 and clean not in acceptable_singles:
        return True

    # Rule 2: Too short
    if len(clean) < min_len:
        return True

    # Rule 3: Known bad name
    if clean in bad_names:
        return True

    # Rule 4: Not snake_case
    if not is_snake_case(func_name):
        return True

    return False


def get_effective_param_count(num_parameter: int,
                              func_params_raw: str) -> int:
    """
    Get the effective parameter count, excluding self and cls.

    Rationale: 'self' and 'cls' are mandatory in instance
    and class methods but carry no design information.
    Counting them would artificially inflate parameter counts
    for all class methods.

    Chen et al. (2018) define Long Parameter List at the
    interface level — parameters the caller must provide.
    """
    if not isinstance(func_params_raw, str):
        return max(0, int(num_parameter or 0))
    try:
        params = json.loads(func_params_raw)
        meaningful = [p for p in params
                      if p not in ("self", "cls")]
        return len(meaningful)
    except (json.JSONDecodeError, ValueError):
        return max(0, int(num_parameter or 0))

# ============================================================
# LABEL GENERATORS
# One function per PDS type, applied row-by-row.
# Each returns a dict of sub-type flags + one master label.
# ============================================================

def label_code_smells(row: pd.Series,
                      nloc_threshold: float) -> dict:
    """
    Detect code smell PDS instances.

    Smells detected:
      - LongMethod      : nloc > 75th percentile (Chen 2018)
      - HighComplexity  : cc > 10 (McCabe 1976)
      - LongParamList   : effective params > 5 (Chen 2018)

    Returns dict with individual flags and master label.
    """
    cc   = int(row.get("cyclomatic_complexity", 0) or 0)
    nloc = int(row.get("nloc", 0) or 0)

    effective_params = get_effective_param_count(
        row.get("num_parameter", 0),
        row.get("function_params", "[]")
    )

    is_long_method   = nloc > nloc_threshold
    is_high_cc       = cc > CC_THRESHOLD
    is_long_params   = effective_params > LONG_PARAM_THRESHOLD

    return {
        "cs_long_method":   int(is_long_method),
        "cs_high_cc":       int(is_high_cc),
        "cs_long_params":   int(is_long_params),
        "code_smell_label": int(
            is_long_method or is_high_cc or is_long_params
        ),
    }


def label_anti_patterns(row: pd.Series,
                        god_class_set: set) -> dict:
    """
    Detect anti-pattern PDS instances.

    Anti-patterns detected:
      - SpaghettiCode : branches > 12 (McCabe 1976/Pylint R0912)
      - HighFanOut    : outgoing calls > 15 (Palomba 2018)

    """
    n_branches = count_branches(
        row.get("function_body_line_type", "{}")
    )
    n_outgoing = float(row.get("outgoing_function_count", 0) or 0)

    is_spaghetti = n_branches > MAX_BRANCHES
    is_high_fanout = n_outgoing > HIGH_FANOUT_THRESHOLD

    return {
        "ap_spaghetti":       int(is_spaghetti),
        "ap_high_fanout":     int(is_high_fanout),
        #"ap_god_class":       int(is_god_class),
        "anti_pattern_label": int(
            is_spaghetti or is_high_fanout
        ),
    }


def label_poor_naming(row: pd.Series) -> dict:
    """
    Detect poor naming convention PDS instances.

    Rules grounded in:
      - Arnaoudova et al. (2016) EMSE Q1
      - PEP 8 (Python official style guide)

    Detection:
      - Single-char names outside the acceptable set
      - Names shorter than MIN_MEANINGFUL_NAME_LEN
      - Names in the known bad patterns list
      - Non-snake_case names
    """
    func_name = str(row.get("function_name", "") or "")

    is_poor = is_poor_name(
        func_name,
        BAD_NAME_PATTERNS,
        ACCEPTABLE_SINGLE_LETTERS,
        MIN_MEANINGFUL_NAME_LEN
    )

    # Additional check: very long names that concatenate
    # many concepts suggest the function does too much
    # (Arnaoudova et al. 2016: identifier bloat)
    is_bloated_name = len(func_name.lstrip("_")) > 50

    return {
        "pn_invalid_name":  int(is_poor),
        "pn_bloated_name":  int(is_bloated_name),
        "poor_naming_label": int(is_poor or is_bloated_name),
    }


def label_poor_documentation(row: pd.Series) -> dict:
    """
    Detect poor documentation PDS instances.

    Taxonomy from Tamrakar et al. (2021) SANER:
      - Lazy    : missing docstring entirely (PEP 257)
      - Trivial : docstring present but < MIN_DOCSTRING_WORDS

    Dunder methods (__init__, __str__, etc.) are exempt from
    mandatory documentation. PEP 257 states these are
    self-documenting by convention, and Pylint's C0116
    exempts them by default. Aligning with this convention
    improves agreement with Pylint and avoids inflating
    the poor documentation rate with unavoidable patterns.

    Private functions (single leading underscore) still
    require documentation — only dunder methods are exempt.
    """
    func_body = str(row.get("function_body", "") or "")
    func_name = str(row.get("function_name", "") or "")

    # Dunder methods are exempt from documentation requirements
    # This aligns with PEP 257 convention and Pylint C0116
    # behaviour — both treat dunders as self-documenting
    is_dunder = (func_name.startswith("__") and
                 func_name.endswith("__"))

    if is_dunder:
        return {
            "pd_missing_doc":  0,
            "pd_trivial_doc":  0,
            "pd_is_dunder":    1,
            "poor_doc_label":  0,
        }

    doc_present = has_docstring(func_body)
    n_doc_words = (count_docstring_words(func_body)
                   if doc_present else 0)

    is_lazy    = not doc_present
    is_trivial = doc_present and n_doc_words < MIN_DOCSTRING_WORDS

    return {
        "pd_missing_doc":  int(is_lazy),
        "pd_trivial_doc":  int(is_trivial),
        "pd_is_dunder":    int(is_dunder),
        "poor_doc_label":  int(is_lazy or is_trivial),
    }

# ============================================================
# GOD CLASS COMPUTATION
#
# God Class is a class-level anti-pattern that cannot be
# detected at the function level in isolation.
# We aggregate function counts per class, then flag
# individual functions that belong to God Classes.
#
# Threshold: > 20 public methods (Pylint R0904 default)
# Justification: Palomba et al. (2018) EMSE Q1
# ============================================================

def compute_god_class_set(df: pd.DataFrame) -> set:
    """
    Identify God Classes in the dataset.

    A God Class is defined as a class with more than
    GOD_CLASS_METHOD_THRESHOLD public methods.

    'Public' means: function name does not start with '_'
    (following Python's convention for private names,
    as defined in PEP 8).

    Returns a set of (project_name, file_path, class_name)
    tuples representing God Class instances.
    """
    # Only consider functions that are inside a class
    # (class_name is not null)
    class_funcs = df[df["class_name"].notna()].copy()

    if len(class_funcs) == 0:
        return set()

    # Mark public functions (not starting with underscore)
    class_funcs["is_public"] = ~class_funcs[
        "function_name"
    ].str.startswith("_", na=False)

    # Count public methods per class
    class_counts = (
        class_funcs[class_funcs["is_public"]]
        .groupby(["project_name", "file_path", "class_name"])
        .size()
        .reset_index(name="public_method_count")
    )

    # Flag classes exceeding the threshold
    god_classes = class_counts[
        class_counts["public_method_count"] >
        GOD_CLASS_METHOD_THRESHOLD
    ]

    # Build set of (project_name, file_path, class_name) tuples
    god_class_set = set(
        zip(
            god_classes["project_name"],
            god_classes["file_path"],
            god_classes["class_name"],
        )
    )

    return god_class_set

# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def process_split(
    df: pd.DataFrame,
    split_name: str,
    nloc_threshold: float,
    god_class_set: set,
) -> pd.DataFrame:
    """
    Apply all PDS labels to one split (train/val/test).
    Returns the dataframe with all label columns added.
    """
    total  = len(df)
    start  = time.time()
    print(f"\nProcessing {split_name}: {total:,} rows...")

    # Apply all four label functions row by row
    # We use a single pass to avoid reading each row 4 times
    results = {
        "cs_long_method":    np.zeros(total, dtype=np.int8),
        "cs_high_cc":        np.zeros(total, dtype=np.int8),
        "cs_long_params":    np.zeros(total, dtype=np.int8),
        "code_smell_label":  np.zeros(total, dtype=np.int8),
        "ap_spaghetti":      np.zeros(total, dtype=np.int8),
        "ap_high_fanout":    np.zeros(total, dtype=np.int8),
        "ap_god_class":      np.zeros(total, dtype=np.int8),
        "anti_pattern_label":np.zeros(total, dtype=np.int8),
        "pn_invalid_name":   np.zeros(total, dtype=np.int8),
        "pn_bloated_name":   np.zeros(total, dtype=np.int8),
        "poor_naming_label": np.zeros(total, dtype=np.int8),
        "pd_missing_doc":    np.zeros(total, dtype=np.int8),
        "pd_trivial_doc":    np.zeros(total, dtype=np.int8),
        "pd_is_dunder":      np.zeros(total, dtype=np.int8),
        "poor_doc_label":    np.zeros(total, dtype=np.int8),
    }

    REPORT_EVERY = 100_000

    for i, (_, row) in enumerate(df.iterrows()):
        cs  = label_code_smells(row, nloc_threshold)
        ap  = label_anti_patterns(row, god_class_set)
        pn  = label_poor_naming(row)
        pd_ = label_poor_documentation(row)

        for k, v in {**cs, **ap, **pn, **pd_}.items():
            results[k][i] = v

        if (i + 1) % REPORT_EVERY == 0:
            elapsed  = time.time() - start
            rate     = (i + 1) / elapsed
            eta_min  = (total - i - 1) / rate / 60
            pct      = (i + 1) / total * 100
            print(f"  [{i+1:>9,}/{total:,}] {pct:>5.1f}%"
                  f"  ETA: {eta_min:.0f}m")

    # Add all label columns to dataframe
    for col, values in results.items():
        df[col] = values

    elapsed = time.time() - start
    print(f"  Completed in {elapsed/60:.1f} minutes")

    return df


def print_label_report(df: pd.DataFrame, split_name: str):
    """
    Print label distribution statistics.
    This is the output you report in your thesis methodology.
    """
    total = len(df)
    print(f"\n{'='*60}")
    print(f"LABEL DISTRIBUTION — {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Total functions: {total:,}")
    print()

    label_groups = {
        "Code Smells": [
            ("cs_long_method",   "Long Method      (Chen et al. 2018)"),
            ("cs_high_cc",       "High Complexity  (McCabe 1976)     "),
            ("cs_long_params",   "Long Param List  (Chen et al. 2018)"),
            ("code_smell_label", "ANY Code Smell   (master label)    "),
        ],
        "Anti-Patterns": [
            ("ap_spaghetti",       "Spaghetti Code   (McCabe 1976)     "),
            ("ap_high_fanout",     "High Fan-Out     (Palomba 2018)    "),
            ("anti_pattern_label", "ANY Anti-Pattern (master label)    "),
        ],
        "Poor Naming": [
            ("pn_invalid_name",  "Invalid Name     (Arnaoudova 2016) "),
            ("pn_bloated_name",  "Bloated Name     (Arnaoudova 2016) "),
            ("poor_naming_label","ANY Poor Naming  (master label)    "),
        ],
        "Poor Documentation": [
            ("pd_missing_doc",   "Missing Docstring (PEP 257/Tamrakar)"),
            ("pd_trivial_doc",   "Trivial Docstring (Tamrakar 2021)  "),
            ("poor_doc_label",   "ANY Poor Doc     (master label)    "),
        ],
    }

    for group_name, labels in label_groups.items():
        print(f"  {group_name}:")
        for col, desc in labels:
            if col in df.columns:
                n   = df[col].sum()
                pct = n / total * 100
                print(f"    {desc}: "
                      f"{n:>9,}  ({pct:>5.1f}%)")
        print()

    # Overall PDS rate
    any_pds = (
        (df["code_smell_label"]   == 1) |
        (df["anti_pattern_label"] == 1) |
        (df["poor_naming_label"]  == 1) |
        (df["poor_doc_label"]     == 1)
    )
    n_any   = any_pds.sum()
    pct_any = n_any / total * 100
    print(f"  Functions with ANY PDS : "
          f"{n_any:,}  ({pct_any:.1f}%)")
    print(f"  Clean functions        : "
          f"{total - n_any:,}  ({100 - pct_any:.1f}%)")

# ============================================================
# MAIN
# ============================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    log_path   = os.path.join(LOG_DIR, "pds_labels_run.log")
    sys.stdout = DualOutput(log_path)

    print("=" * 65)
    print("PRIME — PDS Label Generation")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # ----------------------------------------------------------
    # Step 1: Load training set only for threshold computation
    # The NLOC threshold must be computed from training data
    # only — computing it from all splits would leak test info
    # ----------------------------------------------------------
    print("Loading training set for threshold computation...")
    train_df = pd.read_parquet(
        os.path.join(SPLITS_DIR, "train.parquet")
    )

    print(f"  Train shape: {train_df.shape}")

    # Compute NLOC threshold from training set
    # Chen et al. (2018) statistics-based strategy:
    # threshold = 75th percentile of metric distribution
    nloc_threshold = float(
        np.percentile(train_df["nloc"].dropna(), LONG_METHOD_PERCENTILE)
    )
    print(f"\n  NLOC threshold (75th percentile of train): "
          f"{nloc_threshold:.1f}")
    print(f"  CC threshold                             : "
          f"{CC_THRESHOLD}")
    print(f"  Parameter threshold                      : "
          f"{LONG_PARAM_THRESHOLD}")
    print(f"  Max branches threshold                   : "
          f"{MAX_BRANCHES}")
    print(f"  High fan-out threshold                   : "
          f"{HIGH_FANOUT_THRESHOLD}")
    print(f"  God Class method threshold               : "
          f"{GOD_CLASS_METHOD_THRESHOLD}")
    print(f"  Min docstring words                      : "
          f"{MIN_DOCSTRING_WORDS}")
    print()

    # ----------------------------------------------------------
    # Step 2: Compute God Class set from training data
    # We use only training data to identify God Classes,
    # then apply the same set when labelling val/test
    # ----------------------------------------------------------
    # ============================================================

    splits_to_process = {
        "train": train_df,
        "val":   None,   # will be loaded in loop
        "test":  None,   # will be loaded in loop
    }

    for split_name, df in splits_to_process.items():

        # Load if not already in memory
        if df is None:
            print(f"Loading {split_name} split...")
            df = pd.read_parquet(
                os.path.join(SPLITS_DIR, f"{split_name}.parquet")
            )

        # Apply labels
        output_path = os.path.join(
            OUTPUT_DIR, f"{split_name}_labelled.parquet"
        )
        df = process_split(
            df, split_name, nloc_threshold, set()  # empty set for God Class, as we skip it in anti-patterns
        )

        print_label_report(df, split_name)
        df.to_parquet(output_path, index=False)
        size_gb = os.path.getsize(output_path) / 1e9
        print(f"  Saved to {output_path} ({size_gb:.2f} GB)")

        # Free memory before loading next split
        del df
        import gc
        gc.collect()

    # ----------------------------------------------------------
    # Step 3: Process each split
    # ----------------------------------------------------------
    splits = ["train", "val", "test"]

    print()
    print("=" * 65)
    print("✅ PDS label generation complete.")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()