# PRIME-Py: Python Repository Inspection and Metric Extraction Dataset

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

Replication package for two studies:

> **[Function level]** Alehaidib R, Ghoneim A, Alrashoud M. 2026. Large-Scale Empirical Study
> of Code Smell and Anti-Pattern Detection in Python Open-Source Software.
> *PeerJ Computer Science*. DOI: [paper DOI to be added on acceptance]
> Dataset (v.1.0): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20094326.svg)](https://doi.org/10.5281/zenodo.20094326)

> **[Class level]** Alehaidib R, Ghoneim A, Alrashoud M. 2026. Metric Thresholds for God Class
> and Large Class Detection in Python: A Human-Annotated Validation Study.
> *Software Quality Journal*. DOI: [paper DOI to be added on acceptance]
> Dataset (v.2.0): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21903177.svg)](https://doi.org/10.5281/zenodo.21903177)

---

## Overview

PRIME-Py is a large-scale dataset of **1,997,535 Python functions** extracted
from **2,797 open-source GitHub repositories**, annotated with Poor Design
Symptom (PDS) labels following the empirical research guidelines of
Kitchenham et al. (2002).

The dataset is published at two levels of granularity:

- **Function level** — the base corpus. Five structural PDS sub-types are
  labelled from code metrics alone.
- **Class level** — **derived** from the function level by aggregation, not
  collected separately. Function rows are grouped into class units and two
  class-level symptoms are labelled: God Class and Large Class.

Both levels are published together so that the provenance chain from function
rows to class units remains inspectable and reproducible.

This package addresses the **structural subset of PDS** — Code Smell and
Anti-Pattern. Lexical PDS types (Poor Naming and Poor Documentation) will be
investigated in future work.

### Function-level structural PDS sub-types

| Category | Sub-type | Threshold | Citation |
|---|---|---|---|
| Code Smell | Long Method | NLOC > 14 (75th pct of training corpus) | Chen et al. (2018) IST |
| Code Smell | High Cyclomatic Complexity | CC > 10 | McCabe (1976) IEEE TSE |
| Code Smell | Long Parameter List | params > 5 (non-self) | Chen et al. (2018) IST |
| Anti-Pattern | Spaghetti Code | branch count > 12 | McCabe (1976) / Pylint R0912 |
| Anti-Pattern | High Fan-Out | outgoing calls > 15 | Palomba et al. (2018) EMSE |

### Class-level structural PDS sub-types

| Category | Sub-type | Threshold | Citation |
|---|---|---|---|
| Anti-Pattern | God Class | public methods > 30 | Lanza and Marinescu (2006) |
| Code Smell | Large Class | class NLOC > 54 (75th pct of training corpus) | Alves et al. (2010) |

The Large Class threshold is **derived from this corpus**, not imported from
published values. The 75th-percentile derivation principle is intended to
transfer across corpora; the literal value of 54 is corpus-specific and should
be recomputed for any new corpus rather than reused directly. Class boundaries
are derived from the Python abstract syntax tree rather than from the minimum
function start line, which avoids boundary errors on decorated methods and
nested definitions.

---

## Dataset Files

The dataset files are deposited at Zenodo (open access, CC BY 4.0):

**https://doi.org/10.5281/zenodo.20094326**

### Function level

| File | Rows | Description |
|---|---|---|
| `train_labelled.parquet` | 1,554,896 | Training split — 2,237 projects |
| `val_labelled.parquet` | 165,226 | Validation split — 280 projects |
| `test_labelled.parquet` | 277,413 | Test split — 280 projects |
| `gold_standard.parquet` | 192 | Manually annotated structural PDS subset |

### Class level

| File | Rows | Description |
|---|---|---|
| `train_class_labelled.parquet` | 200,977 | Training split — aggregated from `train_labelled.parquet` |
| `val_class_labelled.parquet` | 19,777 | Validation split — aggregated from `val_labelled.parquet` |
| `test_class_labelled.parquet` | 32,658 | Test split — aggregated from `test_labelled.parquet` |

Total: **253,412 classes**. Project-level split boundaries are inherited from
the function level, so no class appears in more than one split.

### Shared

| File | Description |
|---|---|
| `schema.csv` | Full column schema with types and counting rules |
| `interp1_distribution.csv` | Benchmark validation Interpretation 1 results |
| `interp2_classifier.csv` | Benchmark validation Interpretation 2 results |
| `interp3_direct.csv` | Benchmark validation Interpretation 3 results |

> **Note on the function-level gold standard:** The full annotation study covered
> 384 functions across all four PDS categories (96 per category). This repository
> includes the 192 functions corresponding to the structural PDS scope of the
> function-level paper (Code Smell and Anti-Pattern). The remaining 192 functions
> covering Poor Naming and Poor Documentation will be released with the companion
> lexical PDS study.

> **Note on the class-level gold standard:** The class-level annotation study is a
> separate exercise and does not reuse the function-level annotation. The unit of
> analysis differs, and God Class and Large Class labels are absent from the
> function-level instrument.

### Splits

Splits are performed at the **project level** (seed=42, 80/10/10 ratio).
No function from the same project appears in more than one split, preventing
data leakage between training and evaluation. Class-level splits inherit these
boundaries by construction.

---

## Repository Structure

```
PRIME-Py/
├── README.md
├── annotation/
│   ├── Class_Level/
│   │   ├── adjudication_workbook_71_disagreements.xlsx
│   │   ├── annotator_01.xlsx
│   │   └── annotator_02.xlsx
│   └── Function_Level/
│       ├── annotation_guide.docx
│       ├── annotation_guide.pdf
│       ├── annotation_sample.xlsx
│       └── gold_standard.parquet
├── benchmark_validation/
│   ├── interp1_distribution.csv
│   ├── interp2_classifier.csv
│   └── interp3_direct.csv
├── code/
│   ├── Class_Level/
│   │   ├── class_gold_standard.py
│   │   ├── generate_class_labels.py
│   │   └── match_pylint_to_classes.py
│   ├── Function_Level/
│   │   ├── generate_pds_labels.py
│   │   └── match_pylint_to_functions.py
│   └── requirements.txt
└── data/
    ├── Class_Level/
    │   ├── test_class_labelled.parquet
    │   ├── train_class_labelled.parquet
    │   └── val_class_labelled.parquet
    ├── Function_Level/
    │   ├── test_labelled.parquet
    │   ├── train_labelled.parquet
    │   └── val_labelled.parquet
    └── schema.csv
```

---

## Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/rehaidib/PRIME-Py.git
cd PRIME-Py
```

### 2. Install dependencies

```bash
pip install -r code/requirements.txt
```

### 3. Download the dataset

Download the parquet files from Zenodo:
[https://doi.org/10.5281/zenodo.20094326](https://doi.org/10.5281/zenodo.20094326)

Place them under `data/Function_Level/` and `data/Class_Level/` following the
structure above.

### 4. Reproduce the function-level PDS labels

```bash
python code/Function_Level/generate_pds_labels.py \
  --input_dir data/Function_Level/ \
  --output_dir output/labelled/
```

### 5. Reproduce the function-level Pylint agreement analysis

```bash
python code/Function_Level/match_pylint_to_functions.py \
  --labelled_dir output/labelled/ \
  --pylint_output data/pylint_raw_output.parquet \
  --output_dir output/comparison/
```

### 6. Reproduce the class-level labels

```bash
python code/Class_Level/generate_class_labels.py \
  --input_dir output/labelled/ \
  --output_dir output/class_labelled/
```

This step consumes the function-level labelled output, so step 4 must be run
first. The script aggregates function rows into class units, derives the Large
Class threshold from the training distribution, and applies both class-level
rules. God Class labels are recomputed from source rather than propagated from
any upstream label column.

### 7. Reproduce the class-level Pylint agreement analysis

```bash
python code/Class_Level/match_pylint_to_classes.py \
  --class_dir output/class_labelled/ \
  --output_dir output/class_comparison/
```

### 8. Regenerate the class-level annotation instruments

```bash
python code/Class_Level/class_gold_standard.py \
  --class_dir output/class_labelled/ \
  --output_dir output/annotation/
```

---

## Key Features

### Function level

Each function in the dataset includes:

| Feature Group | Columns | Description |
|---|---|---|
| Identity | `project_name`, `file_path`, `function_name`, `class_name` | Source location |
| Size metrics | `nloc`, `num_token`, `num_parameter` | From Lizard |
| Complexity | `cyclomatic_complexity` | McCabe CC (Lizard) |
| Call graph | `outgoing_function_count`, `incoming_function_count` | From AST |
| Structure | `function_num_variables`, `function_num_functions`, `function_num_lines` | From AST |
| Code Smell labels | `cs_long_method`, `cs_high_cc`, `cs_long_params` | Binary (0/1) |
| Anti-Pattern labels | `ap_spaghetti`, `ap_high_fanout` | Binary (0/1) |
| Master labels | `code_smell_label`, `anti_pattern_label` | Binary (0/1) |
| Semantic | `codet5_summary` | SEBIS CodeTrans T5 summary |

### Class level

Each class in the dataset includes:

| Feature Group | Columns | Description |
|---|---|---|
| Identity | `project_name`, `file_path`, `class_name` | Source location |
| Size metrics | `total_nloc` | Aggregated over class methods |
| Structure | `n_methods`, `n_public`, `n_private` | Method counts by visibility |
| Complexity | cyclomatic complexity statistics | Aggregated over class methods |
| Boundaries | class line span | Derived from AST |
| Labels | God Class, Large Class | Binary (0/1) |

---

## Key Results

### Function level

| Metric | Value |
|---|---|
| Code Smell prevalence (train) | 23.9% |
| Anti-Pattern prevalence (train) | 7.7% |
| Co-occurrence (both categories) | 7.0% |
| Structurally clean functions | 75.4% |
| Long Parameter List — Pylint κ | 0.801 ✅ Almost perfect |
| Spaghetti Code — Pylint κ | 0.837 ✅ Almost perfect |
| Code Smell — gold standard F1 | 0.720 |
| Anti-Pattern — gold standard F1 | 0.567 |
| Long Method — benchmark percentile | 87.6th vs 87.1th (r=0.082) |
| Long Method — benchmark κ (adjusted) | 1.000 (threshold convergence) |

### Class level

Against the 150-class human-annotated gold standard:

| Symptom | Precision | Recall | F1 | κ |
|---|---|---|---|---|
| God Class | 0.760 | 0.826 | 0.792 | 0.694 |
| Large Class | 0.700 | 1.000 | 0.824 | 0.609 |

Benchmarked against Pylint at default settings on the same gold standard:

| Symptom | PRIME-Py F1 | Pylint F1 |
|---|---|---|
| God Class | 0.792 | 0.812 |
| Large Class | 0.824 | 0.462 |

Pylint slightly outperforms on God Class; PRIME-Py substantially outperforms on
Large Class. Both results are reported as observed.

---

## Annotation Study

### Function level

The inter-rater reliability study involved two independent software engineering
practitioners with at least three years of professional Python experience.

| Category | Pre-adjudication κ | Model Precision | Model Recall | Model F1 | Model κ |
|---|---|---|---|---|---|
| Code Smell | 0.060 (slight) | 0.562 | 1.000 | 0.720 | 0.562 |
| Anti-Pattern | 0.167 (slight) | 0.396 | 1.000 | 0.567 | 0.396 |

The low pre-adjudication agreement is reported as a construct validity
observation: threshold-based PDS definitions require explicit quantitative
criteria to achieve practitioner consensus (Yamashita and Counsell, 2013).
See `annotation/Function_Level/annotation_guide.pdf` for the full protocol.

### Class level

A stratified sample of **150 classes** was drawn across six cells, balancing
symptom combinations against distance from the size threshold, with borderline
bands placed around the Large Class cut point so that the sample is not
dominated by clear-cut instances.

| Stratum | n |
|---|---|
| `1_god_large` | 26 |
| `2_large_far` | 25 |
| `3_large_near` | 34 |
| `4_small_near` | 38 |
| `5_small_far` | 25 |
| `6_god_only` | 2 |

Two annotators labelled the sample independently. Annotator sheets present
metric context (NLOC, method counts, thresholds) but **withhold the model's own
verdicts**, so that annotators are not anchored to the labels under evaluation.
The **71 disagreements** were adjudicated by the first author and are recorded
with reasoning in `adjudication_workbook_71_disagreements.xlsx`.

Pre-adjudication agreement is markedly lower for Large Class (κ = 0.230) than
for God Class. This is reported as evidence of construct ambiguity in the Large
Class definition rather than as a weakness of the annotation protocol.

---

## Environment

```
Python          3.11
scikit-learn    1.8.0
scipy           1.17.1
xgboost         3.2.0
pandas          2.2.2
pyarrow         16.1.0
pylint          4.0.5
lizard          1.17.10
```

Full dependency list: see `code/requirements.txt`

---

## Citation

If you use PRIME-Py in your research, please cite the dataset and the relevant
paper:

**Dataset:**
```bibtex
@dataset{alehaidib2026primepy_data,
  title     = {PRIME-Py: Python Repository Inspection and Metric
               Extraction Dataset},
  author    = {Alehaidib, Reem and Ghoneim, Ahmed and Alrashoud, Mubarak},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20094326},
  url       = {https://doi.org/10.5281/zenodo.20094326}
}
```

**Function-level paper:**
```bibtex
@article{alehaidib2026primepy,
  title     = {Large-Scale Empirical Study of Code Smell and Anti-Pattern
               Detection in Python Open-Source Software},
  author    = {Alehaidib, Reem and Ghoneim, Ahmed and Alrashoud, Mubarak},
  journal   = {PeerJ Computer Science},
  year      = {2026},
  doi       = {[paper DOI to be added on acceptance]}
}
```

**Class-level paper:**
```bibtex
@article{alehaidib2026thresholds,
  title     = {Metric Thresholds for God Class and Large Class Detection
               in Python: A Human-Annotated Validation Study},
  author    = {Alehaidib, Reem and Ghoneim, Ahmed and Alrashoud, Mubarak},
  journal   = {Software Quality Journal},
  year      = {2026},
  doi       = {[paper DOI to be added on acceptance]}
}
```

---

## References

- Alves TL, Ypma C, Visser J. 2010. Deriving metric thresholds from benchmark
  data. *ICSM 2010*:1–10.
- Chen Z, et al. 2018. Understanding metric-based detectable smells in
  Python software. *Information and Software Technology* 94:14–29.
- Kitchenham BA, et al. 2002. Preliminary guidelines for empirical research
  in software engineering. *IEEE TSE* 28(8):721–734.
- Lanza M, Marinescu R. 2006. *Object-Oriented Metrics in Practice*. Springer.
- McCabe TJ. 1976. A complexity measure. *IEEE TSE* 2(4):308–320.
- Palomba F, et al. 2018. On the diffuseness and the impact on
  maintainability of code smells. *EMSE* 23(3):1188–1221.
- Sandouka SB, Aljamaan H. 2023. Python code smells detection using
  conventional machine learning models. *PeerJ CS* 9:e1370.
- Yamashita A, Counsell S. 2013. Code smells as system-level indicators
  of maintainability. *JSS* 86(10):2639–2653.

---

## License

- **Dataset** (Zenodo): [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code** (this repository): [MIT License](LICENSE)

---

## Contact

**Reem Alehaidib** (Corresponding Author)
Department of Software Engineering,
College of Computer and Information Sciences,
King Saud University, Riyadh, Saudi Arabia.
Email: [444203308@student.ksu.edu.sa]