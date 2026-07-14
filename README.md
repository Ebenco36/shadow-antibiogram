# Shadow Antibiogram: Co-testing Network Pipeline

This repository contains the code supporting the manuscript:

> **Network-based characterisation of latent co-testing patterns in antimicrobial susceptibility testing**  
> *iScience* (2026) — [https://doi.org/XXXXXXXX](#)

The pipeline constructs **Shadow Antibiograms**: empirical co-testing networks that summarise which antibiotics are tested together in routine microbiology, based on test-incidence (tested vs. not tested) rather than susceptibility outcomes.

---

## Data availability

Processed, non-identifiable data are deposited at Mendeley Data:

> [https://doi.org/10.17632/XXXXXXXX](#) *(DOI to be confirmed upon dataset publication)*

Download the `WHO_Aware_data` folder from that deposit and place it in `datasets/` before running the pipeline (see [Quick start](#quick-start)).

Raw individual-level ARS surveillance records are not included due to data protection and governance constraints.

---

## Quick start

**Prerequisites:** Python 3.x (`python3 --version` to verify). Runs on Linux, macOS, and HPC (SLURM-compatible).

```bash
# 1. Clone the repository
git clone https://github.com/Ebenco36/shadow-antibiogram.git
cd shadow-antibiogram

# 2. Place the deposited WHO_Aware_data folder into datasets/
#    so the path reads: datasets/WHO_Aware_data/

# 3. Run the full pipeline
bash run.sh
```

`run.sh` automatically creates a virtual environment, installs all dependencies from `requirements.txt`, and executes the complete analysis. No manual environment setup is required.

**HPC (SLURM):**
```bash
sbatch --chdir=/path/to/shadow-antibiogram run.sh
```

---

## Outputs

All outputs are written to the `outputs/` directory at runtime:

| Output | Location |
|---|---|
| Similarity matrices and edge lists | `outputs/` |
| FDR-filtered co-testing networks | `outputs/` |
| Louvain community assignments | `outputs/` |
| Manuscript figures | `outputs/use_cases/` |

---

## Repository structure

```
shadow-antibiogram/
├── datasets/
│   └── WHO_Aware_data/        # Aggregated contingency tables (deposit here)
├── src/
│   ├── runners/               # Pipeline orchestration
│   ├── controllers/           # Analysis workflows (AMR, co-testing)
│   ├── mappers/               # Pathogen cohort definitions
│   └── utils/                 # Shared helpers and configuration
├── outputs/                   # Created at runtime
├── run.sh                     # Single entry point
└── requirements.txt
```

---

## Dataset schema

Each file in `datasets/WHO_Aware_data/` contains one row per antibiotic pair per stratum:

| Column | Description |
|---|---|
| `Pathogen` | Full pathogen name (e.g., *Escherichia coli*) |
| `PathogenGenus` | Genus level (e.g., *Escherichia*) |
| `GramType` | Gram classification |
| `Sex` | Patient sex |
| `CareType` | In-Patient / Out-Patient |
| `TextMaterialgroupRkiL0` | Specimen type (e.g., Urine, Blood Culture) |
| `ARS_WardType` | Ward type (Normal Ward, ICU, etc.) |
| `BroadAgeGroup` | Age group category |
| `HighLevelAgeRange` | Detailed age range |
| `Hospital_Priority` | Hospital level (High / Medium / Low) |
| `Care_Complexity` | Care complexity tier |
| `Year` | Year of isolation |
| `ab_1` | First antibiotic |
| `ab_2` | Second antibiotic |
| `a` | Isolates where **both** antibiotics were tested |
| `b` | Isolates where **only ab_1** was tested |
| `c` | Isolates where **only ab_2** was tested |
| `d` | Isolates where **neither** antibiotic was tested |

---

## Analysis configuration

Default settings used in the manuscript:

| Parameter | Value |
|---|---|
| Similarity metric | Jaccard |
| Edge threshold | τ = 0.30 |
| Statistical filter | Fisher's exact test, BH-FDR α = 0.05 |
| Community detection | Louvain, γ = 1.0 |

---

## Interpretation

- Shadow Antibiograms represent **co-testing behaviour**, not resistance outcomes.
- Louvain communities are disjoint by design and reflect dominant co-testing configurations within a cohort, not exclusive biological categories.
- Cross-context comparisons should use fixed parameters (same metric, threshold, and resolution).

---

## Issues

Please open a GitHub issue for questions or problems with the pipeline.
