# Shadow Antibiogram: Co-testing Network Pipeline

This repository implements a co-testing analysis pipeline to construct **Shadow Antibiograms**: empirical networks that summarise **which antibiotics are tested together** in routine microbiology workflows.  
The analysis is based on **test-incidence** (tested vs not tested), rather than susceptibility outcomes, and is designed to characterise diagnostic ordering behaviour across pathogens, specimen types, clinical contexts, and time.

The pipeline is intentionally simple to run and reproducible, with a single entry script that bootstraps the environment and executes all configured analyses.

---

## Getting the code

Clone the repository to a local machine or an HPC-accessible directory:

```bash
git clone https://github.com/Ebenco36/shadow-antibiogram.git
cd shadow-antibiogram
```

If you already have a local copy, ensure it is up to date:

```bash
git pull
```

## OR

```bash
pip install shadow-antibiogram
```


All commands below assume execution from the repository root directory.

---

## Data availability and setup

The dataset used in this study is now included directly in this repository under `datasets/WHO_Aware_data/`.  
To protect patient privacy while preserving the ability to reproduce all analyses, we have **aggregated** the original isolate-level data into contingency tables of co-testing patterns within predefined strata. The aggregation process preserves all pairwise co-testing relationships required for network construction and has been validated against the full dataset to ensure no loss of analytical quality.

**The pipeline is designed to work with both the original complete (isolate-level) dataset and the aggregated version.** For public release and replication, we provide the aggregated data; however, if you have access to the original isolate-level data format, the pipeline can process it directly.

### Dataset contents (aggregated)

The `datasets/WHO_Aware_data/` directory contains one or more CSV files with the following structure (one row per antibiotic pair per stratum):

| Column                | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `Pathogen`            | Full pathogen name (e.g., *Escherichia coli*)                                                 |
| `PathogenGenus`       | Genus level (e.g., *Escherichia*)                                                             |
| `GramType`            | Gram classification (Positive / Negative / etc.)                                              |
| `Sex`                 | Patient sex (if applicable)                                                                   |
| `CareType`            | In-Patient / Out-Patient / etc.                                                               |
| `TextMaterialgroupRkiL0` | Specimen type (e.g., Urine, Blood Culture)                                                  |
| `ARS_WardType`        | Ward type (Normal Ward, ICU, etc.)                                                            |
| `BroadAgeGroup`       | Age group category                                                                            |
| `HighLevelAgeRange`   | Detailed age range                                                                            |
| `Hospital_Priority`   | Hospital level (High / Medium / Low)                                                          |
| `Care_Complexity`     | Complexity of care (Tertiary & Specialized, Primary/Secondary, etc.)                          |
| `Year`                | Year of isolation                                                                             |
| `ab_1`                | First antibiotic (full name with `_Tested` suffix)                                            |
| `ab_2`                | Second antibiotic                                                                             |
| `a`                   | Count of isolates where **both** antibiotics were tested                                      |
| `b`                   | Count where **only ab_1** was tested (ab_1 tested, ab_2 not)                                  |
| `c`                   | Count where **only ab_2** was tested (ab_1 not, ab_2 tested)                                  |
| `d`                   | Count where **neither** antibiotic was tested                                                 |

All analyses reported in the manuscript can be reproduced using these aggregated tables. No additional data preprocessing is required before running the pipeline.

> **Note:** The original isolate-level data are not included due to privacy constraints, but the aggregated format captures all necessary information for co-testing network analysis and has been validated to yield identical results. If you have your own isolate-level data, the pipeline can accept it with minimal adaptation (see [Data requirements](#data-requirements)).

---

### From original to aggregated: an illustration

To make the aggregation process transparent, consider a small hypothetical set of original isolate-level records for *E. coli* from urine specimens in 2022. Each row represents one isolate, with binary indicators showing which antibiotics were tested.

**Original isolate-level data (simplified):**

| Pathogen  | Year | Specimen | Ampicillin_Tested | Gentamicin_Tested | Ciprofloxacin_Tested |
|-----------|------|----------|-------------------|-------------------|----------------------|
| E. coli   | 2022 | Urine    | 1                 | 1                 | 0                    |
| E. coli   | 2022 | Urine    | 1                 | 0                 | 1                    |
| E. coli   | 2022 | Urine    | 0                 | 1                 | 1                    |

For each antibiotic pair within the stratum defined by `Pathogen = E. coli`, `Year = 2022`, and `Specimen = Urine`, we count isolates in four categories:

- **a** = both tested
- **b** = only first tested
- **c** = only second tested
- **d** = neither tested

**Aggregated contingency table for this stratum:**

| ab_1                     | ab_2                     | a | b | c | d |
|--------------------------|--------------------------|---|---|---|---|
| Ampicillin_Tested        | Gentamicin_Tested        | 1 | 1 | 1 | 0 |
| Ampicillin_Tested        | Ciprofloxacin_Tested     | 1 | 1 | 1 | 0 |
| Gentamicin_Tested        | Ciprofloxacin_Tested     | 1 | 1 | 1 | 0 |

*Explanation for the first row (Ampicillin vs. Gentamicin):*  
- Isolate 1 tested both → contributes to **a**  
- Isolate 2 tested only Ampicillin → contributes to **b**  
- Isolate 3 tested only Gentamicin → contributes to **c**  
- No isolate tested neither → **d = 0**

The full dataset includes many such strata (by additional variables like age group, ward type, etc.), and for each stratum every antibiotic pair that appears in at least one isolate is represented. These counts are sufficient to reconstruct all pairwise association measures (Jaccard, Phi, etc.) and to perform statistical tests exactly as if the original isolate-level data were available.

---

## Overview of the approach

The pipeline is designed to handle both **complete (isolate-level)** and **aggregated** datasets seamlessly. Given input data with:

* organism metadata (genus and, when available, species),
* contextual metadata (specimen type, care setting, ward, year),
* antibiotic testing information (which antibiotics were tested for each isolate),

it performs the following steps.

---

### 1. Test-incidence encoding

Diagnostic behaviour is encoded as a binary **test-incidence matrix** with dimensions:

* N isolates × A antibiotics

where:

* `1` indicates that an antibiotic was tested for a given isolate,
* `0` indicates that it was not tested.

This representation isolates **diagnostic ordering decisions** and does not depend on resistance or susceptibility results.

When aggregated data are provided, the loader reconstructs the necessary internal structures directly from the contingency tables.

---

### 2. Pairwise co-testing association

Within a defined cohort (for example, organism × specimen × year), the pipeline computes pairwise co-testing frequency and binary association measures for all antibiotic pairs.

Supported similarity metrics include:

* **Jaccard** (default): overlap among tested sets; well suited to panel-like behaviour,
* Dice and Cosine: proportional overlap measures,
* Phi: sensitive to joint absences and often yields fragmented structure.

---

### 3. Network construction

Antibiotics are represented as nodes in a weighted, undirected graph.  
Edges connect antibiotic pairs whose association exceeds a user-defined threshold (`tau`).  
Edge weights store the similarity value.

Each network represents observed co-testing behaviour within a specific analytical cohort.

---

### 4. Optional statistical validation

Edges above the similarity threshold can be validated using **Fisher’s exact test** applied to the 2×2 contingency table for each antibiotic pair.  
Multiple testing is controlled using **Benjamini–Hochberg false discovery rate (FDR)** correction.

When enabled, only statistically supported edges are retained.

---

### 5. Community detection (optional)

**Louvain modularity optimisation** is used to summarise networks into **disjoint communities**, interpreted as **dominant co-testing configurations** within a cohort (for example, organism × specimen × year).

Because Louvain produces non-overlapping partitions, communities should be interpreted as descriptive summaries of prevailing structure rather than as overlapping or exhaustive diagnostic panels.

---

### 6. Outputs

Depending on the enabled modules, the pipeline produces:

* co-testing networks (GraphML, GEXF, or CSV edge lists),
* node and edge summary tables,
* temporal testing-volume analyses,
* organisation participation summaries,
* manuscript-ready figures.

---

## One-command execution (recommended)

### Local machine

```bash
bash run.sh
```

### HPC (Slurm)

```bash
sbatch --chdir=/path/to/your/repo run.sh
```

The `--chdir` flag ensures execution from a writable directory rather than a Slurm spool location.

---

## What `run.sh` does

The script automatically:

1. Sets a safe working directory.
2. Creates or reuses a Python virtual environment:  
   * `./.venv` if the repository is writable,  
   * otherwise `~/.venvs/<repo_name>`.
3. Installs dependencies from `requirements.txt`.
4. Executes the configured analysis modules sequentially.

No manual environment setup is required.

---

## Repository structure (overview)

```text
shadow-antibiogram/
├── datasets/
│   └── WHO_Aware_data/          # Aggregated contingency tables (CSV)
│       ├── aggregated_data_1.csv
│       └── ...
├── output/                       # Pipeline outputs (created at runtime)
├── src/
│   ├── runners/                  # Pipeline entry points and orchestration
│   ├── controllers/              # Analysis workflows (e.g. AMR, co-testing)
│   ├── data/                     # Data loading and preprocessing utilities
│   ├── networks/                 # Network construction and analysis
│   └── utils/                    # Shared helpers and configuration
│   └── Preprocessing/            # Directory for data preprocessing.
├── results/                      # Output for sensitivity analysis results
├── results_use_cases/            # Output for use cases
├── temporal_analysis/            # Output for temporal analysis results
├── run.sh
├── requirements.txt
└── README.md
```

The primary user-facing entry point is `run.sh`. Users do not need to invoke Python modules directly unless performing custom analyses.

---

## Data requirements

The pipeline includes a flexible data loader that accepts either:

* **Complete isolate-level data** – a dataframe with at least:  
  * a temporal field (`Year` or date),  
  * an organisation identifier (if available),  
  * organism labels (genus or species),  
  * specimen type,  
  * antibiotic testing information (where any recorded value indicates "tested", missing indicates "not tested").

* **Aggregated contingency tables** (as provided in `datasets/WHO_Aware_data/`) – the loader reconstructs the necessary internal structures directly from the counts.

If antibiotic data are recorded as susceptibility results, any recorded value (S, I, R, or MIC) is treated as **tested**, while missing values indicate **not tested**.

---

## Interpretation notes

* Shadow Antibiograms represent **co-testing behaviour**, not resistance.
* Louvain communities are **disjoint by design** and should be interpreted as dominant configurations within a cohort, not as exclusive biological categories.
* Comparisons across time or clinical context should be made **under fixed parameters** (same similarity metric, thresholds, and resolution) to avoid methodological artefacts.

---

## Configuration (typical defaults)

The following settings are aligned with the manuscript analyses:

* Similarity metric: `jaccard`
* Edge threshold: `tau = 0.30`
* Statistical validation: Fisher’s exact test with BH-FDR at `alpha = 0.05`
* Louvain resolution: `gamma = 1.0`
* Repeated initialisations for stability assessment (if enabled)

---

## Reproducibility

For reproducible runs:

* fix random seeds where applicable,
* log the configuration used for each analysis,
* record software versions (for example, using `pip freeze`).

---

## Citation and attribution

If this code is used in a manuscript, relevant methodological references include:

* WHO AWaRe classification for antibiotic annotation,
* STL or additive decomposition methods for temporal analyses,
* Louvain modularity optimisation for community detection,
* Benjamini–Hochberg FDR correction for multiple testing.

---

This README provides all necessary information to understand, run, and reproduce the Shadow Antibiogram pipeline using either the provided aggregated dataset or your own complete isolate-level data format. For any questions or issues, please open an issue on GitHub.