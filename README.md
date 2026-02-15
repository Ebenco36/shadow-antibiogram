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

All commands below assume execution from the repository root directory.

---

## Data availability and setup

The dataset used in this study is publicly available via **Zenodo**.

**Zenodo record**
DOI: `10.5281/zenodo.18274234`
File: `WHO_Aware_data.tar.xz`

The archive contains preprocessed antimicrobial susceptibility testing metadata required to reproduce all analyses reported in the manuscript. No patient-identifiable data are included.

### Downloading the dataset

From the repository root directory:

```bash
mkdir -p datasets && \
wget -O datasets/WHO_Aware_data.tar.xz \
"https://zenodo.org/records/18274234/files/WHO_Aware_data.tar.xz?download=1" && \
tar -xf datasets/WHO_Aware_data.tar.xz -C . --strip-components=1 && \
rm datasets/WHO_Aware_data.tar.xz
```

Alternatively, using `curl`:

```bash
mkdir -p datasets && \
curl -L -o datasets/WHO_Aware_data.tar.xz \
"https://zenodo.org/records/18274234/files/WHO_Aware_data.tar.xz?download=1" && \
tar -xf datasets/WHO_Aware_data.tar.xz -C . --strip-components=1 && \
rm datasets/WHO_Aware_data.tar.xz
```

After extraction, the expected path is:

```text
datasets/WHO_Aware_data/
```

No additional data preprocessing is required.

---

## Overview of the approach

Given a surveillance dataset of clinical isolates with:

* organism metadata (genus and, when available, species),
* contextual metadata (specimen type, care setting, ward, year),
* antibiotic testing information (which antibiotics were tested for each isolate),

the pipeline performs the following steps.

---

### 1. Test-incidence encoding

Diagnostic behaviour is encoded as a binary **test-incidence matrix** with dimensions:

* N isolates × A antibiotics

where:

* `1` indicates that an antibiotic was tested for a given isolate,
* `0` indicates that it was not tested.

This representation isolates **diagnostic ordering decisions** and does not depend on resistance or susceptibility results.

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
│   └── WHO_Aware_data/
    └── output/
├── src/
│   ├── runners/            # Pipeline entry points and orchestration
│   ├── controllers/        # Analysis workflows (e.g. AMR, co-testing)
│   ├── data/               # Data loading and preprocessing utilities
│   ├── networks/           # Network construction and analysis
│   └── utils/              # Shared helpers and configuration
│   └── Preprocessing/      # Directory for data preprocessing.
├── results/                # Output for sensitivity analysis results
├── results_use_cases/      # Output for use cases
├── temporal_analysis/      # Output for temporal analysis results
├── run.sh
├── requirements.txt
└── README.md
```

The primary user-facing entry point is `run.sh`. Users do not need to invoke Python modules directly unless performing custom analyses.

---

## Data requirements

The data loader must return a dataframe containing at least:

* a temporal field (`Year` or date),
* an organisation identifier,
* organism labels (genus or species),
* specimen type,
* antibiotic testing information.

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

