# ============================================================
# match_pylint_to_functions.py
# PRIME Dataset — Match Pylint Output to Function-Level Labels
#
# PURPOSE:
#   Matches Pylint messages to specific functions in your
#   labelled dataset using file_path and line number ranges.
#   Produces a comparison dataset with both metric-based
#   labels and Pylint-based labels for agreement analysis.
#
# MATCHING RULE:
#   A Pylint message at line L belongs to function F if:
#     F.start_line <= L <= F.end_line
#
#   For God Class messages (R0902, R0904) which report at
#   the class definition line rather than inside a method,
#   we match to all functions belonging to that class in
#   the same file.
#
# OUTPUT:
#   comparison_dataset.parquet
#   Contains all labelled functions with additional columns:
#     pylint_poor_doc      : 1 if C0116 found in function
#     pylint_poor_naming   : 1 if C0103 found in function
#     pylint_spaghetti     : 1 if R0912 found in function
#     pylint_long_params   : 1 if R0913 found in function
#     pylint_god_class     : 1 if R0902 or R0904 in class
#
# HOW TO RUN:
#   conda activate prime
#   python match_pylint_to_functions.py
# ============================================================

import os
import sys
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

PYLINT_RAW_PATH  = "/Users/reemehaidib/PhD_Dataset/PRIME_Output/pylint/pylint_raw_output.parquet"
LABELLED_DIR     = "/Users/reemehaidib/PhD_Dataset/PRIME_Output/labelled"
MASTER_PATH      = "/Users/reemehaidib/PhD_Dataset/PRIME_Output/master_dataset_with_lexical.parquet"
OUTPUT_DIR       = "/Users/reemehaidib/PhD_Dataset/PRIME_Output/comparison"
LOG_DIR          = "/Users/reemehaidib/PhD_Dataset/PRIME_Logs"

# Pylint message ID to PDS type mapping
FUNCTION_LEVEL_IDS = {
    "C0116": "pylint_poor_doc",      # missing-function-docstring
    "C0103": "pylint_poor_naming",   # invalid-name
    "R0912": "pylint_spaghetti",     # too-many-branches
    "R0913": "pylint_long_params",   # too-many-arguments
}

# Class-level IDs — matched differently (see below)
CLASS_LEVEL_IDS = {
    "R0902": "pylint_god_class",     # too-many-instance-attributes
    "R0904": "pylint_god_class",     # too-many-public-methods
}

# ============================================================
# LOAD AND FILTER
# ============================================================

def load_and_filter_pylint(master_projects: set) -> pd.DataFrame:
    """
    Load Pylint raw output and filter to only the 2,797
    projects in the master dataset.

    This removes the 45 projects that were cloned but
    produced no extractable functions in Lizard/AST.
    """
    print("Loading Pylint raw output...")
    pylint_df = pd.read_parquet(PYLINT_RAW_PATH)
    print(f"  Raw shape          : {pylint_df.shape}")
    print(f"  Unique projects    : "
          f"{pylint_df['project_name'].nunique():,}")

    # Filter to master dataset projects
    before = len(pylint_df)
    pylint_df = pylint_df[
        pylint_df["project_name"].isin(master_projects)
    ].copy()
    after = len(pylint_df)

    print(f"  After filtering    : {after:,} messages "
          f"(removed {before - after:,} from "
          f"non-master projects)")
    print(f"  Unique projects    : "
          f"{pylint_df['project_name'].nunique():,}")
    print()

    return pylint_df


def load_labelled_splits() -> pd.DataFrame:
    """
    Load and concatenate all three labelled splits.
    We work on the full dataset for the comparison,
    then the notebook will analyse by split if needed.
    """
    print("Loading labelled splits...")
    dfs = []
    for split in ["train", "val", "test"]:
        path = os.path.join(
            LABELLED_DIR, f"{split}_labelled.parquet"
        )
        df   = pd.read_parquet(path)
        df["split"] = split
        dfs.append(df)
        print(f"  {split:5s}: {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total : {len(combined):,} rows")
    print()
    return combined

# ============================================================
# MATCHING LOGIC
#
# Two separate matching strategies for function-level
# and class-level Pylint messages.
# ============================================================

def build_function_level_index(
    pylint_df: pd.DataFrame,
) -> dict:
    """
    Build a lookup structure for function-level messages.

    Structure:
      {file_path: [(line, message_id), ...]}

    This allows O(1) file lookup followed by a linear
    scan over messages in that file for each function.
    """
    func_msgs = pylint_df[
        pylint_df["message_id"].isin(FUNCTION_LEVEL_IDS.keys())
    ].copy()

    index = {}
    for _, row in func_msgs.iterrows():
        fp   = row["file_path"]
        line = int(row["line"])
        mid  = row["message_id"]
        if fp not in index:
            index[fp] = []
        index[fp].append((line, mid))

    print(f"  Function-level index: "
          f"{len(index):,} files with messages")
    return index


def build_class_level_index(
    pylint_df: pd.DataFrame,
) -> dict:
    """
    Build a lookup structure for class-level messages.

    God Class messages (R0902, R0904) are reported at the
    class definition line, not inside any function.
    We match them to all functions in the same file that
    share the same class_name.

    Structure:
      {(file_path, class_obj_name): True}

    class_obj_name is the 'obj' field from Pylint output,
    which contains the class name for R0902/R0904 messages.
    """
    cls_msgs = pylint_df[
        pylint_df["message_id"].isin(CLASS_LEVEL_IDS.keys())
    ].copy()

    index = set()
    for _, row in cls_msgs.iterrows():
        fp  = row["file_path"]
        obj = str(row["obj"] or "")
        if obj:
            index.add((fp, obj))

    print(f"  Class-level index : "
          f"{len(index):,} (file, class) pairs with messages")
    return index


def match_pylint_to_functions(
    labelled_df: pd.DataFrame,
    func_index:  dict,
    class_index: set,
) -> pd.DataFrame:
    """
    Match Pylint messages to each function using line ranges.

    For function-level messages:
      A message at line L belongs to function F if
      F.start_line <= L <= F.end_line

    For class-level messages (God Class):
      All functions in the same (file_path, class_name)
      are flagged if that class has a God Class message.

    Returns labelled_df with new pylint_* columns added.
    """
    total = len(labelled_df)
    start = time.time()
    print(f"Matching Pylint messages to {total:,} functions...")

    # Initialise all Pylint label columns to 0
    pylint_cols = list(FUNCTION_LEVEL_IDS.values()) + ["pylint_god_class"]
    pylint_cols = list(set(pylint_cols))  # deduplicate
    for col in pylint_cols:
        labelled_df[col] = np.int8(0)

    REPORT_EVERY = 200_000

    for i, (idx, row) in enumerate(labelled_df.iterrows()):
        file_path  = row["file_path"]
        start_line = int(row.get("start_line", 0) or 0)
        end_line   = int(row.get("end_line",   0) or 0)
        class_name = str(row.get("class_name", "") or "")

        # --- Function-level matching ---
        if file_path in func_index:
            for msg_line, msg_id in func_index[file_path]:
                if start_line <= msg_line <= end_line:
                    col = FUNCTION_LEVEL_IDS[msg_id]
                    labelled_df.at[idx, col] = 1

        # --- Class-level matching (God Class) ---
        if class_name and (file_path, class_name) in class_index:
            labelled_df.at[idx, "pylint_god_class"] = 1

        if (i + 1) % REPORT_EVERY == 0:
            elapsed  = time.time() - start
            rate     = (i + 1) / elapsed
            eta_min  = (total - i - 1) / rate / 60
            pct      = (i + 1) / total * 100
            print(f"  [{i+1:>9,}/{total:,}]  "
                  f"{pct:>5.1f}%  ETA: {eta_min:.0f}m")

    elapsed = time.time() - start
    print(f"  Matching complete in {elapsed/60:.1f} minutes")
    return labelled_df

# ============================================================
# AGREEMENT ANALYSIS
#
# Computes precision, recall, F1, and Cohen's Kappa for
# each PDS type comparing metric-based vs Pylint labels.
#
# Why Cohen's Kappa and not just accuracy?
# Accuracy is misleading when classes are imbalanced.
# If 80% of functions have missing docs, a detector that
# always says "yes" achieves 80% accuracy trivially.
# Kappa corrects for chance agreement, which is the
# standard metric in empirical software engineering
# inter-rater agreement studies.
# ============================================================

def cohens_kappa(y_true: pd.Series,
                 y_pred: pd.Series) -> float:
    """
    Compute Cohen's Kappa between two binary label series.
    Kappa > 0.6 = substantial agreement (publishable)
    Kappa > 0.8 = almost perfect agreement
    """
    n  = len(y_true)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = (y_true == y_pred).mean()

    # Expected agreement by chance
    p_yes_true = y_true.mean()
    p_yes_pred = y_pred.mean()
    p_e = (p_yes_true * p_yes_pred +
           (1 - p_yes_true) * (1 - p_yes_pred))

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def compute_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute agreement metrics for each comparable PDS pair.

    Comparable pairs:
      Metric cs_long_params  vs Pylint pylint_long_params
      Metric ap_spaghetti    vs Pylint pylint_spaghetti
      Metric poor_naming_label vs Pylint pylint_poor_naming
      Metric poor_doc_label  vs Pylint pylint_poor_doc

    Note on naming comparison scope:
      Your metric-based detector checks function NAMES only.
      Pylint C0103 checks ALL identifiers in the file.
      We therefore restrict the naming comparison to
      function-level matches only (where Pylint's obj field
      contains the function name) — but since we already
      did the match at line level, we report it as-is
      and document the scope difference.
    """
    comparisons = [
        {
            "pds_type":    "Long Parameter List",
            "citation":    "Chen et al. (2018) IST",
            "metric_col":  "cs_long_params",
            "pylint_col":  "pylint_long_params",
            "scope_note":  "Function-level, threshold > 5 args",
        },
        {
            "pds_type":    "Spaghetti Code",
            "citation":    "McCabe (1976) IEEE TSE",
            "metric_col":  "ap_spaghetti",
            "pylint_col":  "pylint_spaghetti",
            "scope_note":  "Function-level, threshold > 12 branches",
        },
        {
            "pds_type":    "Poor Naming",
            "citation":    "Arnaoudova et al. (2016) EMSE",
            "metric_col":  "poor_naming_label",
            "pylint_col":  "pylint_poor_naming",
            "scope_note":  "Metric=func names only; "
                           "Pylint=all identifiers in file",
        },
        {
            "pds_type":    "Poor Documentation",
            "citation":    "Tamrakar et al. (2021) SANER",
            "metric_col":  "poor_doc_label",
            "pylint_col":  "pylint_poor_doc",
            "scope_note":  "Pylint exempts dunder methods",
        },
        {
            "pds_type":    "God Class",
            "citation":    "Palomba et al. (2018) EMSE",
            "metric_col":  "ap_god_class",
            "pylint_col":  "pylint_god_class",
            "scope_note":  "Class-level aggregation",
        },
    ]

    results = []
    for comp in comparisons:
        mc  = comp["metric_col"]
        pc  = comp["pylint_col"]

        if mc not in df.columns or pc not in df.columns:
            continue

        y_metric = df[mc].fillna(0).astype(int)
        y_pylint = df[pc].fillna(0).astype(int)

        # Confusion matrix components
        tp = int(((y_metric == 1) & (y_pylint == 1)).sum())
        fp = int(((y_metric == 1) & (y_pylint == 0)).sum())
        fn = int(((y_metric == 0) & (y_pylint == 1)).sum())
        tn = int(((y_metric == 0) & (y_pylint == 0)).sum())

        n         = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall /
                     (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / n if n > 0 else 0.0
        kappa     = cohens_kappa(y_metric, y_pylint)

        # Label rates
        metric_rate = y_metric.mean() * 100
        pylint_rate = y_pylint.mean() * 100

        results.append({
            "PDS Type":       comp["pds_type"],
            "Citation":       comp["citation"],
            "N Functions":    n,
            "Metric Rate %":  round(metric_rate, 1),
            "Pylint Rate %":  round(pylint_rate, 1),
            "TP":             tp,
            "FP":             fp,
            "FN":             fn,
            "TN":             tn,
            "Precision":      round(precision, 3),
            "Recall":         round(recall, 3),
            "F1":             round(f1, 3),
            "Accuracy":       round(accuracy, 3),
            "Cohen Kappa":    round(kappa, 3),
            "Scope Note":     comp["scope_note"],
        })

    return pd.DataFrame(results)


def print_agreement_report(agreement_df: pd.DataFrame):
    """Print the agreement report in thesis-ready format."""
    print()
    print("=" * 70)
    print("AGREEMENT ANALYSIS — Metric-Based vs Pylint Labels")
    print("=" * 70)
    print("Interpretation guide:")
    print("  Kappa > 0.80 = Almost perfect agreement")
    print("  Kappa > 0.60 = Substantial agreement (publishable)")
    print("  Kappa > 0.40 = Moderate agreement")
    print("  Kappa < 0.40 = Fair/poor agreement — needs discussion")
    print()

    for _, row in agreement_df.iterrows():
        print(f"  {row['PDS Type']}")
        print(f"    Citation       : {row['Citation']}")
        print(f"    N Functions    : {row['N Functions']:,}")
        print(f"    Metric rate    : {row['Metric Rate %']}%")
        print(f"    Pylint rate    : {row['Pylint Rate %']}%")
        print(f"    Precision      : {row['Precision']}")
        print(f"    Recall         : {row['Recall']}")
        print(f"    F1             : {row['F1']}")
        print(f"    Cohen's Kappa  : {row['Cohen Kappa']} "
              f"{'✅ Substantial' if row['Cohen Kappa'] >= 0.6 else '⚠️  Moderate' if row['Cohen Kappa'] >= 0.4 else '❌ Fair/Poor'}")
        print(f"    Scope note     : {row['Scope Note']}")
        print()

# ============================================================
# MAIN
# ============================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    log_path   = os.path.join(LOG_DIR, "match_pylint.log")
    sys.stdout = DualOutput(log_path)

    print("=" * 65)
    print("PRIME — Match Pylint to Function-Level Labels")
    print(f"Started : "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # -------------------------------------------------------
    # Step 1: Load master project list for filtering
    # -------------------------------------------------------
    print("Loading master project list...")
    master_projects = set(
        pd.read_parquet(
            MASTER_PATH, columns=["project_name"]
        )["project_name"].unique()
    )
    print(f"  Master dataset projects: {len(master_projects):,}")
    print()

    # -------------------------------------------------------
    # Step 2: Load and filter Pylint output
    # -------------------------------------------------------
    pylint_df = load_and_filter_pylint(master_projects)

    # -------------------------------------------------------
    # Step 3: Load labelled splits
    # -------------------------------------------------------
    labelled_df = load_labelled_splits()

    # -------------------------------------------------------
    # Step 4: Build matching indexes
    # -------------------------------------------------------
    print("Building matching indexes...")
    func_index  = build_function_level_index(pylint_df)
    class_index = build_class_level_index(pylint_df)
    print()

    # Free Pylint dataframe — no longer needed after indexing
    del pylint_df
    import gc
    gc.collect()

    # -------------------------------------------------------
    # Step 5: Match Pylint messages to functions
    # -------------------------------------------------------
    labelled_df = match_pylint_to_functions(
        labelled_df, func_index, class_index
    )

    # -------------------------------------------------------
    # Step 6: Compute and print agreement analysis
    # -------------------------------------------------------
    print("\nComputing agreement analysis...")
    agreement_df = compute_agreement(labelled_df)
    print_agreement_report(agreement_df)

    # -------------------------------------------------------
    # Step 7: Save outputs
    # -------------------------------------------------------
    comparison_path = os.path.join(
        OUTPUT_DIR, "comparison_dataset.parquet"
    )
    agreement_path  = os.path.join(
        OUTPUT_DIR, "agreement_results.csv"
    )

    print("Saving outputs...")
    labelled_df.to_parquet(comparison_path, index=False)
    agreement_df.to_csv(agreement_path, index=False)

    size_gb = os.path.getsize(comparison_path) / 1e9
    print(f"  Comparison dataset : {comparison_path} "
          f"({size_gb:.2f} GB)")
    print(f"  Agreement results  : {agreement_path}")
    print()
    print("=" * 65)
    print("✅ Matching and agreement analysis complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()