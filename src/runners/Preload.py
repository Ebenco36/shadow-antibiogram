from pathlib import Path
import pandas as pd

from src.RawDataLoader import load_data_feather, parquet_ready
from src.Preprocessing.DataPreprocessor import DataPreprocessor
from src.mappers.pathogen_names import pathogen_name_maps_dict
from src.Preprocessing.DataPreprocessorCleaning import DataPreprocessorCleaning
from src.utils.helpers import merge_antibiotic_data


if __name__ == "__main__":
    # 1) Ensure base parquet dataset exists (created by your loader)
    if not parquet_ready():
        print("Parquet not found — generating from feather chunks...")
        load_data_feather()

    # 2) Merge antibiotic metadata (kept at the old path your code expects)
    merge_antibiotic_data(
        existing_file="./datasets/antibiotic_classification_complete_original.csv",
        who_file="./datasets/AWaRe Classification 2023.csv",
        output_file="./datasets/antibiotic_classification_complete.csv",
    )

    # 3) Read the big base dataset from the parquet folder produced by the loader
    base_parquet_dir = Path("./datasets/dataset_parquet/")
    data = pd.read_parquet(base_parquet_dir, engine="pyarrow")

    # 4) Enrichment / renaming
    preprocessor = DataPreprocessor(data)
    df_mapping = pd.read_csv("./datasets/antibiotic_classification_complete.csv")

    preprocessor.rename_columns_from_mapping(
        df_mapping, key_col="Antibiotic Name German", value_col="Full Name"
    )

    material_dict = {
        "Urin": "Urine", "Wunde": "Wound", "Abstrich": "Swab", "respiratorisch": "Respiratory",
        "Blutkultur": "Blood Culture", "Stuhlproben": "Stool Samples", "Urogenitalabstrich": "Urogenital Swab",
        "Punktat": "Punctate", "sonstiges": "Other", "Ausschluss": "Exclusion", "unbekannt": "Unknown",
    }
    preprocessor.replace_values("TextMaterialgroupRkiL0", material_dict)

    region_dict = {
        "West": "West", "Südost": "South East", "Nordost": "North East", "Südwest": "Southwest",
        "Nordwest": "Northwest", "unbekannt": "Unknown",
    }
    preprocessor.replace_values("ARS_Region", region_dict)

    hospital_level_dict = {
        "Level 1 - Grundversorgung": "Level 1 - Basic Care",
        "Level 2 - Regelversorgung": "Level 2 - Regular Care",
        "Level 3 - Schwerpunktversorgung": "Level 3 - Specialized Care",
        "Level 4 - Maximalversorgung": "Level 4 - Maximum Care",
        "Level 5 - Fachkrankenhäuser": "Level 5 - Specialized Hospitals",
        "Level 6 - Sonstige Krankenhäuser": "Level 6 - Other Hospitals",
        "Level 7 - Vorsorge- und Rehabilitationseinrichtung": "Level 7 - Preventive and Rehabilitation Facilities",
        "nicht zugeordnet": "Not Assigned",
    }
    preprocessor.replace_values("ARS_HospitalLevelManual", hospital_level_dict)

    ward_type_dict = {
        "Normalstation": "Normal Ward", "ambulant": "Outpatient", "Intensivstation": "Intensive Care Unit",
        "sonstige Behandlungsart": "Other Treatment Type", "Intermediate Care/Wachstation": "Intermediate Care/Awake Station",
        "Rehabilitation": "Rehabilitation", "OP": "Operating Room", "Frührehabilitation": "Early Rehabilitation",
        "unbekannt": "Unknown", "Tagesklinik": "Day Clinic",
    }
    preprocessor.replace_values("ARS_WardType", ward_type_dict)

    pathogen_group_dict = {
        "gram-negativ: Enterobacterales": "Gram-negative: Enterobacterales",
        "gram-positiv: Streptokokken": "Gram-positive: Streptococci",
        "gram-positiv: Staphylokokken": "Gram-positive: Staphylococci",
        "gram-negativ: Nonfermenter": "Gram-negative: Non-fermenters",
        "gram-positiv: andere": "Gram-positive: Other",
        "Anaerobier": "Anaerobes",
        "gram-negativ: andere": "Gram-negative: Other",
        "gram-positiv: Enterokokken": "Gram-positive: Enterococci",
        "sonstige": "Miscellaneous",
    }
    enriched = preprocessor.replace_values("PathogengroupL1", pathogen_group_dict)

    col = enriched["PathogengroupL1"].astype("string").str.strip()
    neg = col.str.startswith("Gram-negative", na=False)
    pos = col.str.startswith("Gram-positive", na=False)
    ana = col.eq("Anaerobes").fillna(False)
    misc = col.eq("Miscellaneous").fillna(False)

    enriched["GramType"] = pd.Series(pd.NA, index=enriched.index, dtype="string")
    enriched.loc[neg, "GramType"] = "Gram-negative"
    enriched.loc[pos, "GramType"] = "Gram-positive"
    enriched.loc[ana, "GramType"] = "Anaerobes"
    enriched.loc[misc, "GramType"] = "Miscellaneous"

    enriched["PathogenSummary"] = enriched["PathogengroupL1"].str.extract(r":\s*(.*)")

    preprocessor.replace_values("Pathogen", pathogen_name_maps_dict)

    # 5) Save the enriched INTERIM table (no version folder)
    saver = DataPreprocessorCleaning(csv_path="", cat_columns=[])
    interim_path = saver.save_data(
        df=enriched,
        name="CompleteData_en",
        stage="interim",
        format="parquet",
        compression="zstd",
        partition_on=None,        # <- key change
        rows_per_file=200_000,    # split into multiple files in one dir
    )
    print(f"Saved interim parquet to: {interim_path}")  # -> data/interim/CompleteData_en/dataset_parquet