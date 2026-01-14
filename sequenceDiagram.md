sequenceDiagram
    %% Force larger text by styling each participant

    participant RawData as Raw Data Layer <br> (ARS/LIS Records)
    participant DataPrep as Data Extraction & <br> Preparation Module
    participant CleanedData as Cleaned & Prepared <br> Dataset
    participant Antibiogram as Antibiogram Matrix (0/1)
    participant Context as Context Features



    RawData->>DataPrep: Input All Records

    activate DataPrep
        %% Note over DataPrep: Data Cleaning & Anonymization<br>• Remove identifiers<br>• Anonymize demographics<br>• Harmonize metadata
        Note over DataPrep: Dataset & Feature Filtering<br>• Exclude complex antibiotics<br>• Filter low-coverage (<1%)<br>• Deduplicate isolates
        Note over DataPrep: Feature Engineering & Target Creation<br>• Binary indicator (Tested/Not)<br>• Add antibiotic class, AWaRe<br>• Add pathogen taxonomy
    deactivate DataPrep

    DataPrep->>Antibiogram: Output Processed Antibiogram Data
    DataPrep->>Context: Output Enriched Context Features

    Antibiogram-->>CleanedData: Available for Analysis
    Context-->>CleanedData: Available for Analysis

    Note over CleanedData: Ready for Cohort Selection & Downstream Analysis
