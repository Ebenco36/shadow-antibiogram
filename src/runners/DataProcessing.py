# src/controllers/DataProcessing.py

from src.runners.DataDescriptor import run_full_descriptor
from src.mappers.top_pathogens import CRITICAL_PATHOGENS, HIGH_PATHOGENS, MEDIUM_PATHOGENS
from src.controllers.DatabaseOverview import run_database_stats
from src.utils.LoadClasses import LoadClasses
from src.mappers.exempted_columns import columns_to_exempt
from src.utils.helpers import build_broad_class_map, build_class_map, build_who_map, compute_row_features, pick_abx_cols, prepare_feature_inputs, save_json

import pandas as pd
import numpy as np
import altair as alt
import sys, os, re, uuid, shutil, json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join('.')))
alt.data_transformers.enable('default')

# ======================
# Parquet-friendly I/O
# ======================

_UNNAMED_RE = re.compile(r"^Unnamed(?::\s*\d+)?$")

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if _UNNAMED_RE.match(str(c))]
    return df.drop(columns=cols) if cols else df

def _iter_row_chunks(df: pd.DataFrame, rows_per_file: int):
    n = len(df)
    for start in range(0, n, rows_per_file):
        yield df.iloc[start:start + rows_per_file]

def save_parquet_flat(
    df: pd.DataFrame, out_dir: Path,
    *, rows_per_file: int = 250_000, compression: str = "zstd"
) -> Path:
    """
    Save a DataFrame as a *flat* Parquet dataset (no partition subfolders):
      out_dir/part-00000.parquet, part-00001.parquet, ...
    """
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    tmp = out_dir.parent / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=True)

    pdf = _drop_unnamed(df)
    file_idx = 0
    for chunk in _iter_row_chunks(pdf, rows_per_file):
        (tmp / f"part-{file_idx:05d}.parquet").parent.mkdir(parents=True, exist_ok=True)
        chunk.to_parquet(
            tmp / f"part-{file_idx:05d}.parquet",
            engine="pyarrow",
            compression=compression,
            index=False,
        )
        file_idx += 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.move(str(tmp), str(out_dir))

    meta = {
        "format": "parquet",
        "layout": "flat",
        "files": file_idx,
        "rows": int(pdf.shape[0]),
        "cols": int(pdf.shape[1]),
        "compression": compression,
    }
    with open(out_dir / "_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir

def read_any(path: str | Path) -> pd.DataFrame:
    """Auto-detect: Parquet dir/file, Feather, or CSV."""
    p = Path(path)
    if p.is_dir():
        return pd.read_parquet(p, engine="pyarrow")
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p, engine="pyarrow")
    if suf in (".feather", ".ft"):
        return pd.read_feather(p)
    return pd.read_csv(p, low_memory=False)


# ======================
# DataProcessing
# ======================

class DataProcessing:
    def __init__(
        self, file_path,
        output_path: str = "./datasets/output",
        population_file: str = "./datasets/population-data-cleaned.csv",
        antibiotics_class_file: str = "./datasets/antibiotic_classification_complete.csv",
        outcome_scheme: str = "R_vs_SI",   # "RI_vs_S" | "R_vs_SI" | "ordinal_R2I1S0"
        outcome_dtype: str = "Int8"        # pandas nullable integer
    ):
        """
        file_path (str | Path): Path to the dataset. Prefer a Parquet *directory*.
        """
        self.load = LoadClasses()
        self.file_path = file_path
        self.output_path = output_path
        self.population_file = population_file
        self.outcome_scheme = outcome_scheme
        self.outcome_dtype = outcome_dtype

        _, self.charts_dir, self.tables_dir = self.setup_directories(self.output_path)
        self.antibiotic_class_data = self.load.load_antibiotic_classification(
            antibiotic_classification_file=antibiotics_class_file
        )

        # Load and enrich
        self.data = self.load_data()
        self.population_data = self.load_population_data()

        run_full_descriptor(self.data, name="AMR_2025Q1_COMPLETE", out_root="reports/CompleteData")

        # Strict: only RAW antibiotic result columns (contain " - " and have NO derived suffix)
        self.antibiotic_columns = [
            col for col in self.data.columns
            if (" - " in col)
            and not col.endswith("_Tested")
            and not col.endswith("_Outcome")
            and col not in columns_to_exempt
        ]

        # One-time cleanup for any legacy artifacts
        artifact_cols = [c for c in self.data.columns if c.endswith("_Tested_Outcome")]
        if artifact_cols:
            print(f"[CLEANUP] Dropping {len(artifact_cols)} artifact columns (*_Tested_Outcome).")
            self.data.drop(columns=artifact_cols, inplace=True, errors="ignore")

        self.preprocess_data()

    # ---------- setup / IO ----------

    def setup_directories(self, output_path, charts_dir_="charts", tables_dir_="tables"):
        def ensure_directory(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            print(f"Directory '{directory_path}' is ready.")

        ensure_directory(output_path)
        charts_dir = os.path.join(output_path, charts_dir_)
        ensure_directory(charts_dir)
        tables_dir = os.path.join(output_path, tables_dir_)
        ensure_directory(tables_dir)

        _, self.charts_dir, self.tables_dir = (output_path, charts_dir, tables_dir)
        return output_path, charts_dir, tables_dir

    def load_population_data(self):
        df = pd.read_csv(self.population_file)
        df["Bundesland"] = df["bundesland"].str.strip()
        df["Year"] = df["Year"].astype(str)
        df["total"] = df["total"].astype(float)
        return df[["Bundesland", "Year", "total"]]

    def load_data(self):
        """
        Load the dataset (Parquet dir/file preferred; CSV/Feather also supported).
        """
        df = read_any(self.file_path)

        # Feature engineering (matching your prior code)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["MonthName"] = df["Date"].dt.month_name()
        df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
        df['SeasonCode'] = df['Month'] % 12 // 3 + 1
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        df['SeasonName'] = df['SeasonCode'].map(season_map)

        df['hospital_level_group'] = np.where(
            df['ARS_HospitalLevelManual'].astype('string').str.contains(r'\bLevel\s*[3-7]\b', na=False),
            'Specialized Care', 'Basic Care'
        )

        # Identify raw antibiotic columns (at this stage may still include derived in the source; we filter later)
        antibiotic_columns = [c for c in df.columns if ' - ' in c and c not in columns_to_exempt]

        df['TotalAntibioticsTested'] = df[antibiotic_columns].notna().sum(axis=1)

        if "Year" in df.columns:
            df["Year"] = df["Year"].astype(str)

        # IMPORTANT: prevent helpers from mutating raw R/I/S columns → pass a copy
        helper_inputs_df = df.copy(deep=True)
        antibiotics, class_map, who_class_map = prepare_feature_inputs(helper_inputs_df, self.antibiotic_class_data)
        # compute_row_features uses row from original df (read-only); dicts map from helper copy
        feature_df = df.apply(
            compute_row_features,
            axis=1,
            result_type="expand",
            args=(antibiotics, class_map, who_class_map)
        )
        feature_df.columns = ['NumDrugClassesTested', 'IsCriticalDrugTested', 'BackupDrugsTested']
        df = pd.concat([df, feature_df], axis=1)

        df = self.enrich_data(df)

        # Drop columns that are entirely NaN (never reported)
        zero_cols = [col for col in antibiotic_columns if df[col].notna().sum() == 0]
        with open(f"{self.output_path}/zero_tested_antibiotics.txt", "w", encoding="utf-8") as f:
            for col in zero_cols:
                f.write(f"{col}\n")
        if zero_cols:
            df.drop(columns=zero_cols, inplace=True)

        return df

    # ---------- R/I/S normalization & outcome encoding ----------

    @staticmethod
    def _norm_result_token(x):
        import numpy as _np, re as _re
        if x is None or (isinstance(x, float) and _np.isnan(x)): return _np.nan
        s = str(x).strip().lower()
        if s in {"", ".", "-", "na", "nan", "none", "missing", "null"}: return _np.nan
        if s in {"r","res","resistant"}: return "R"
        if s in {"s","sus","susceptible"}: return "S"
        if s in {"i","int","intermediate"}: return "I"
        # optional numeric fallbacks if present in source data
        if s == "1": return "R"
        if s == "0": return "S"
        if "/" in s or "\\" in s:
            parts = {p.strip() for p in _re.split(r"[\\/]", s)}
            if "r" in parts: return "R"
            if "i" in parts: return "I"
            if "s" in parts: return "S"
            return _np.nan
        if s and s[0] in {"r","i","s"}: return s[0].upper()
        return _np.nan

    @staticmethod
    def _encode_outcome(tokens: pd.Series, scheme: str, dtype: str = "Int8") -> pd.Series:
        """
        One outcome series for one antibiotic.
        NaN tokens remain <NA> (pandas nullable Int8).
        """
        if scheme == "RI_vs_S":
            out = pd.Series(pd.NA, index=tokens.index, dtype=dtype)
            out = out.mask(tokens.isna(), pd.NA)
            out = out.where(~tokens.isin(["R", "I"]), 1)
            out = out.where(~(tokens == "S"), 0)
            return out.astype(dtype)

        if scheme == "R_vs_SI":
            out = pd.Series(pd.NA, index=tokens.index, dtype=dtype)
            out = out.mask(tokens.isna(), pd.NA)
            out = out.where(~(tokens == "R"), 1)
            out = out.where(~tokens.isin(["S", "I"]), 0)
            return out.astype(dtype)

        if scheme == "ordinal_R2I1S0":
            mapping = {"R": 2, "I": 1, "S": 0}
            return tokens.map(mapping).astype(dtype)

        raise ValueError(f"Unknown outcome scheme: {scheme}")

    # ---------- core processing ----------

    def preprocess_data(self):
        print(f"[INFO] Building _Tested and _Outcome (scheme: {self.outcome_scheme})")

        # Normalize minimal missing markers (do NOT coerce raw R/I/S to numeric)
        self.data[self.antibiotic_columns] = self.data[self.antibiotic_columns].replace(
            ["", " ", "NA", "-", "."], np.nan
        )

        # Hard guard: ensure only raw base columns here
        for col in self.antibiotic_columns:
            if col.endswith("_Tested") or col.endswith("_Outcome"):
                raise RuntimeError(f"Derived column slipped into antibiotic_columns: {col}")

        # Build _Tested flags
        tested_df = self.data[self.antibiotic_columns].notna().astype("Int8")
        tested_df.columns = [f"{c}_Tested" for c in self.antibiotic_columns]

        # Build _Outcome with selected scheme (preserves <NA> for untested)
        outcome_cols = {}
        for col in self.antibiotic_columns:
            tokens = self.data[col].apply(self._norm_result_token)  # does NOT modify originals
            outcome_cols[f"{col}_Outcome"] = self._encode_outcome(
                tokens, scheme=self.outcome_scheme, dtype=self.outcome_dtype
            )
        outcome_df = pd.DataFrame(outcome_cols, index=self.data.index)

        # Assemble
        meta_cols = [
            "NumberOrganisation", "OrgType", "Anonymized_Lab", "IdPatientDW", "Pathogen", "PathogengroupL1", 
            "GramType", "Sex", "Date", "PathogenGenus",
            "TextMaterialgroupRkiL0", "AgeGroup", "AgeRange", "CareType", "ARS_HospitalLevelManual",
            "ARS_WardType", "ARS_Region", "Year", "Month", "MonthName", "YearMonth", "SeasonCode",
            "SeasonName", "TotalAntibioticsTested", "PathogenSummary", "BroadAgeGroup",
            "HighLevelAgeRange", "Hospital_Priority", "Care_Complexity", "Facility_Function",
            "CSQ", "CSQMG", "CSY", "CSYMG", "IsSpecificlyExcluded_Screening", "TypeOrganisation",
            "IsSpecificlyExcluded_Pathogen", "IsSpecificlyExcluded_PathogenevidenceNegative"
        ]
        meta_present = [c for c in meta_cols if c in self.data.columns]

        self.data = pd.concat([self.data[meta_present], tested_df, outcome_df], axis=1)

        # Keep isolates with at least one test
        self.data = self.filter_isolates_with_tests(self.data)

        # Sanity check: outcome must be NaN where Tested==0
        outcome_cols_now = [c for c in self.data.columns if c.endswith("_Outcome") and " - " in c]
        for ycol in outcome_cols_now:
            base = ycol[:-8]  # strip "_Outcome"
            tcol = f"{base}_Tested"
            if tcol in self.data.columns:
                bad = self.data.loc[(self.data[tcol] == 0) & self.data[ycol].notna(), [tcol, ycol]]
                if not bad.empty:
                    raise AssertionError(f"{base}: found non-NA outcome where Tested==0")

        # Drop low-coverage antibiotics by _Tested
        kept_by_tested, test_stats, affected_patients = self.drop_low_tested_antibiotics(
            self.data, min_tested_ratio=0.01, suffix="_Tested", patient_id_col="IdPatientDW", verbose=True
        )
        self.plot_kept_vs_dropped_altair(test_stats)
        self.data = kept_by_tested

        # Drop low-coverage outcomes by presence (not-NA)
        self.data, outcome_stats, outcome_affected = self.drop_low_tested_antibiotics(
            self.data, min_tested_ratio=0.01, suffix="_Outcome", patient_id_col="IdPatientDW", verbose=True
        )

        # Drop IdPatientDW after filtering/exports prepared
        self.data = self.data.drop(columns=["IdPatientDW"], errors="ignore")

        # Save flat parquet dataset
        out_dir = Path("./datasets") / "structured/dataset_parquet"
        save_parquet_flat(self.data, out_dir, rows_per_file=250_000, compression="zstd")
        run_full_descriptor(
            self.data, name="AMR_2025Q1_COMPLETE_STRUCTURED_FILTERED",
            out_root="reports/CompleteData_Structured_Filtered"
        )
        print(f"[OK] Saved Parquet dataset to: {out_dir}")

        # Save unique lists
        columns_to_save = [
            ('TextMaterialgroupRkiL0', 'UniqueMaterials'),
            ('Pathogen', 'UniquePathogens'),
            ('PathogenGenus', 'UniquePathogenGenus'),
            ('ARS_Region', 'UniqueARS_Region'),
            ('ARS_WardType', 'UniqueARS_WardType'),
            ('ARS_HospitalLevelManual', 'UniqueARS_HospitalLevelManual'),
            ('TypeOrganisation', 'UniqueTypeOrganisation'),
            ('CareType', 'UniqueCareType'),
            ('PathogengroupL1', 'UniquePathogengroupL1'),
            ('GramType', 'UniqueGramType'),
            ('Sex', 'UniqueSex'),
            ('AgeGroup', 'UniqueAgeGroup'),
            ('AgeRange', 'UniqueAgeRange'),
            ('BroadAgeGroup', 'UniqueBroadAgeGroup'),
            ('HighLevelAgeRange', 'UniqueHighLevelAgeRange'),
            ('Hospital_Priority', 'UniqueHospital_Priority'),
            ('Care_Complexity', 'UniqueCare_Complexity'),
            ('Facility_Function', 'UniqueFacility_Function'),
            ('Year', 'UniqueYear'),
        ]
        Path("./datasets/unique_columns").mkdir(parents=True, exist_ok=True)
        for column, filename in columns_to_save:
            try:
                vals = pd.Series(self.data[column].unique(), name=filename)
                vals.to_csv(f"./datasets/unique_columns/{filename}.csv", index=False)
            except KeyError as e:
                print(f"Column {column} not found in the data: {e}")
            except Exception as e:
                print(f"Error while processing column {column}: {e}")

    # ---------- droppers / plots ----------

    def drop_low_tested_antibiotics(
        self, df, min_tested_ratio=0.01, suffix="_Tested",
        patient_id_col="IdPatientDW", verbose=True
    ):
        num_rows = len(df)
        test_cols = [col for col in df.columns if col.endswith(suffix)]

        if suffix == "_Outcome":
            # Presence = notna() → 1, else 0
            present = df[test_cols].notna().astype(int)
        else:
            # For _Tested, columns are 0/1 Int8 already
            present = df[test_cols].astype(float)

        test_counts = present.sum()
        test_means = present.mean()

        test_stats = pd.DataFrame({
            "Tested Count": test_counts.astype(int),
            "Tested %": (test_means * 100).round(2)
        })
        test_stats["Status"] = test_means.apply(lambda x: "Kept" if x >= min_tested_ratio else "Dropped")

        kept_cols = test_stats[test_stats["Status"] == "Kept"].index.tolist()
        dropped_cols = test_stats[test_stats["Status"] == "Dropped"].index.tolist()
        meta_cols = [col for col in df.columns if col not in test_cols]

        df_filtered = df[meta_cols + kept_cols]
        affected_mask = df[dropped_cols].notna().any(axis=1) if suffix == "_Outcome" else (df[dropped_cols].sum(axis=1) > 0)
        affected_patients = df.loc[affected_mask]

        if verbose:
            print(f"Total rows: {num_rows}")
            print(f"Threshold: {min_tested_ratio*100:.1f}% of patients")
            print(f"Kept {len(kept_cols)} antibiotic columns ({suffix})")
            print(f"Dropped {len(dropped_cols)} antibiotic columns ({suffix})")
            kept = test_stats[test_stats["Status"] == "Kept"].sort_values(by="Tested Count", ascending=False)
            print(" Top 10 Most Present:")
            print(kept.head(10))
            kept.to_csv(f"{self.output_path}/cleaned_kept_columns{suffix}.csv", index=True)
            dropped = test_stats[test_stats["Status"] == "Dropped"].sort_values(by="Tested Count")
            print("\n Dropped (low presence):")
            print(dropped.head(10))
            dropped.to_csv(f"{self.output_path}/dropped_columns{suffix}.csv", index=True)
            affected_patients.to_csv(f"{self.output_path}/affected_patients{suffix}.csv", index=False)

        return df_filtered, test_stats, affected_patients

    def plot_kept_vs_dropped_altair(self, test_stats: pd.DataFrame):
        if "Status" not in test_stats.columns:
            raise ValueError("`test_stats` must contain a 'Status' column.")
        df = test_stats["Status"].value_counts().reset_index()
        df.columns = ["Status", "Count"]
        df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int)
        df["Percent"] = (df["Count"] / df["Count"].sum() * 100).round(1)
        df["Label"] = df["Status"] + " (" + df["Percent"].astype(str) + "%)"

        color_scale = alt.Scale(domain=["Kept", "Dropped"], range=["#4E79A7", "#F28E2B"])
        base = alt.Chart(df).encode(theta=alt.Theta("Count:Q", stack=True),
                                    color=alt.Color("Status:N", scale=color_scale))
        chart = (base.mark_arc(innerRadius=60, outerRadius=210) +
                 base.mark_text(radius=260, size=14, color="black").encode(text="Label:N")).properties(
            width=550, height=550, title="Proportion of Antibiotic Test Columns: Kept vs Dropped"
        )
        chart.save(f"{self.output_path}/kept_vs_dropped_piechart.html")
        return chart

    # ---------- misc enrichers ----------

    def assign_pathogen_priority(self, row: pd.Series) -> str:
        text = f"{row.get('Pathogen', '')} {row.get('PathogenGenus', '')} {row.get('PathogenSummary', '')}".lower()
        if re.search(CRITICAL_PATHOGENS, text, flags=re.IGNORECASE):
            return "Critical"
        elif re.search(HIGH_PATHOGENS, text, flags=re.IGNORECASE):
            return "High"
        elif re.search(MEDIUM_PATHOGENS, text, flags=re.IGNORECASE):
            return "Medium"
        else:
            return "Other"

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["PatientAge"] = df["PatientAge"].str.extract(r"(\d+)").astype(float)
        no_age_df = df[df["PatientAge"].isna()]
        no_age_df.to_csv("records_without_age.csv", index=False)
        print(f"{df['PatientAge'].isna().sum()} records have missing PatientAge.")
        df[df["PatientAge"] < 1].to_csv("records_with_age_less_than_1.csv", index=False)
        df["PatientAge"] = df["PatientAge"].fillna(-1).astype(int)

        bins = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 120]
        labels = ['0 years','1-4 years','5-9 years','10-14 years','15-19 years','20-24 years','25-29 years',
                  '30-34 years','35-39 years','40-44 years','45-49 years','50-54 years','55-59 years','60-64 years',
                  '65-69 years','70-74 years','75-79 years','80-84 years','85-89 years','90-94 years','≥95 years']
        df["AgeRange"] = pd.cut(df["PatientAge"], bins=bins, labels=labels, right=False)

        broad_group_map = {'0 years':'Infants','1-4 years':'Early Childhood','5-9 years':'Late Childhood',
                           '10-14 years':'Late Childhood','15-19 years':'Adolescents & Young Adults',
                           '20-24 years':'Adolescents & Young Adults','25-29 years':'Early Middle Age',
                           '30-34 years':'Early Middle Age','35-39 years':'Early Middle Age',
                           '40-44 years':'Late Middle Age','45-49 years':'Late Middle Age',
                           '50-54 years':'Late Middle Age','55-59 years':'Pre-Senior',
                           '60-64 years':'Pre-Senior','65-69 years':'Early Elderly','70-74 years':'Early Elderly',
                           '75-79 years':'Early Elderly','80-84 years':'Late Elderly','85-89 years':'Late Elderly',
                           '90-94 years':'Late Elderly','≥95 years':'Late Elderly'}
        high_level_age_range_map = {
            '0 years':'<15','1-4 years':'<15','5-9 years':'<15','10-14 years':'<15',
            '15-19 years':'15–29','20-24 years':'15–29','25-29 years':'15–29',
            '30-34 years':'30–44','35-39 years':'30–44','40-44 years':'30–44',
            '45-49 years':'45–59','50-54 years':'45–59','55-59 years':'45–59',
            '60-64 years':'60–74','65-69 years':'60–74','70-74 years':'60–74',
            '75-79 years':'75–89','80-84 years':'75–89','85-89 years':'75–89',
            '90-94 years':'90+','≥95 years':'90+'
        }

        df['HighLevelAgeRange'] = df['AgeRange'].map(high_level_age_range_map)
        df['BroadAgeGroup'] = df['AgeRange'].map(broad_group_map)
        df['AgeGroup'] = df['BroadAgeGroup'].map(lambda x: 'Adult' if x in
                                                 ['Early Middle Age','Late Middle Age','Pre-Senior']
                                                 else ('Pediatric' if x in ['Infants','Early Childhood','Late Childhood']
                                                       else ('Elderly' if x in ['Early Elderly','Late Elderly'] else x)))

        df["PathogenPriority"] = df.apply(self.assign_pathogen_priority, axis=1)
        sex_mapping = {"w":"Woman","m":"Man","u":"Others","o":"Others","d":"Others"}
        df["Sex"] = df["Sex"].map(sex_mapping).fillna("Others")

        if "TextMaterialgroupRkiL0" in df.columns:
            material_mapping = {
                'Urine':'Urine','Wound':'Wound','Swab':'Swab','Blood Culture':'Blood Culture','Stool Samples':'Stool Samples',
                'Urogenital Swab':'Urogenital Swab','Respiratory':'Respiratory','Other':'Others','Punctate':'Punctate',
                'Exclusion':'Exclusion','Unknown':'Others'
            }
            df["TextMaterialgroupRkiL0"] = df["TextMaterialgroupRkiL0"].map(material_mapping).fillna("Others")

        if "CareType" in df.columns:
            df["CareType"] = df["CareType"].map({'ambulant':'Out-Patient','stationär':'In-Patient'}).fillna("Others")

        LEVEL_COL = "ARS_HospitalLevelManual"
        care_complexity_map = {
            'Level 1 - Basic Care':'Primary/Secondary',
            'Level 2 - Regular Care':'Primary/Secondary',
            'Level 3 - Specialized Care':'Tertiary & Specialized',
            'Level 4 - Maximum Care':'Tertiary & Specialized',
            'Level 5 - Specialized Hospitals':'Tertiary & Specialized',
            'Level 6 - Other Hospitals':'Other/Peripheral',
            'Level 7 - Preventive and Rehabilitation Facilities':'Preventive/Rehabilitation',
            'Not Assigned':'Unknown','':'Unknown',
        }
        facility_function_map = {
            'Level 1 - Basic Care':'General Hospital',
            'Level 2 - Regular Care':'General Hospital',
            'Level 3 - Specialized Care':'Referral/Advanced Hospital',
            'Level 4 - Maximum Care':'Referral/Advanced Hospital',
            'Level 5 - Specialized Hospitals':'Specialized Facility',
            'Level 6 - Other Hospitals':'Other Hospital',
            'Level 7 - Preventive and Rehabilitation Facilities':'Preventive/Rehab Facility',
            'Not Assigned':'Unclassified','':'Unclassified',
        }
        priority_map = {
            'Level 1 - Basic Care':'Medium','Level 2 - Regular Care':'Medium',
            'Level 3 - Specialized Care':'High','Level 4 - Maximum Care':'High','Level 5 - Specialized Hospitals':'High',
            'Level 6 - Other Hospitals':'Low','Level 7 - Preventive and Rehabilitation Facilities':'Low',
            'Not Assigned':'Unclassified','':'Unclassified',
        }
        
        org_type = {
            'Krankenhaus':'Hospital',
            'Arztpraxis':"Doctor's office",
            'Andere':'Other',
        }

        df['Care_Complexity'] = pd.Categorical(
            df[LEVEL_COL].map(care_complexity_map).fillna('Unknown'),
            categories=['Primary/Secondary','Tertiary & Specialized','Other/Peripheral',
                        'Preventive/Rehabilitation','Unknown'], ordered=True)
        df['Facility_Function'] = pd.Categorical(
            df[LEVEL_COL].map(facility_function_map).fillna('Unclassified'),
            categories=['General Hospital','Referral/Advanced Hospital','Specialized Facility',
                        'Other Hospital','Preventive/Rehab Facility','Unclassified'], ordered=True)
        df['Hospital_Priority'] = pd.Categorical(
            df[LEVEL_COL].map(priority_map).fillna('Unclassified'),
            categories=['High','Medium','Low','Unclassified'], ordered=True)
        
        df['OrgType'] = pd.Categorical(
            df["TypeOrganisation"].map(org_type).fillna('Other'),
            categories=['Hospital',"Doctor's office",'Other'], ordered=True)
        
        # Get unique lab names
        unique_labs = df["ARS_LabName"].unique()
        # Build mapping: real name -> "Lab 1", "Lab 2", ...
        mapping = {name: f"Lab {i+1}" for i, name in enumerate(unique_labs)}
        df["Anonymized_Lab"] = df["ARS_LabName"].map(mapping)

        return df

    def filter_isolates_with_tests(self, data: pd.DataFrame) -> pd.DataFrame:
        test_columns = [col for col in data.columns if col.endswith('_Tested')]
        return data[data[test_columns].sum(axis=1) > 0]

    def generateJson(self, df:pd.DataFrame = None):
        abx_cols_all = pick_abx_cols(df)
        BROAD_JSON = "./datasets/antibiotic_broad_class_grouping.json"
        CLASS_JSON = "./datasets/antibiotic_class_grouping.json"
        WHO_JSON = "./datasets/antibiotic_class.json"

        # 1) Build class map (deduped) from a single list of class names
        CLASS_NAMES = [
            "Fluoroquinolone", "Aminoglycoside", "Penicillin (β-lactam)",
            "β-lactam/β-lactamase inhibitor", "Monobactam (β-lactam)",
            "Third-gen cephalosporin (β-lactam)", "Fourth-gen cephalosporin (β-lactam)",
            "First-gen cephalosporin (β-lactam)", "Pleuromutilin",
            "Siderophore cephalosporin (β-lactam)", "Tetracycline derivative",
            "Macrocyclic", "Aminocyclitol", "Amphenicol", "Tetracycline",
            "Lincosamide", "Fifth-gen cephalosporin (β-lactam)", "Polymyxin",
            "Glycopeptide", "Streptogramin", "Lipopeptide", "Lipoglycopeptide",
            "Phosphonic acid derivative", "Second-gen cephalosporin (β-lactam)",
            "Oxazolidinone", "Macrolide", "Nitrofuran", "Rifamycin", "Glycylcycline",
            "Carbapenem (β-lactam)", "Steroid antibiotic", "Pseudomonic acid",
            "Sulfonamide/Trimethoprim combo", "Quinolone", "Sulfonamide",
            "Dihydrofolate reductase inhibitor", "Polyene", "Polypeptide",
            "Echinocandin", "Azole", "Oxacephem (β-lactam)", "Nitroimidazole",
            "Quinolone derivative", "Streptogramin combo", "Antimetabolite", "Other Class",
            "Ketolide (macrolide derivative)", "Carbacephem (β-lactam)", "β-lactamase inhibitor",
        ]
        BROAD_CLASS_NAMES = [
            "β-lactam",
            "Aminoglycoside",
            "Fluoroquinolone",
            "Tetracycline",
            "Macrolide",
            "Lincosamide",
            "Glycopeptide",
            "Lipopeptide",
            "Oxazolidinone",
            "Amphenicol",
            "Polymyxin",
            "Nitrofuran",
            "Rifamycin",
            "Glycylcycline",
            "Dihydrofolate reductase inhibitor",
            "Steroid antibiotic",
            "Other"
        ]

        CLASS_NAMES = list(dict.fromkeys(CLASS_NAMES))  # dedupe, preserve order
        class_map_present = build_class_map(self.load, df, CLASS_NAMES)
        save_json(class_map_present, CLASS_JSON)
        print("Saved antibiotic classes:", CLASS_JSON)

        # 2) Build WHO map
        who_map_present = build_who_map(self.load, df, ["Watch", "Access", "Reserve", "Not Set"])
        save_json(who_map_present, WHO_JSON)
        print("Saved WHO categories:", WHO_JSON)
        
        broad_map_present = build_broad_class_map(self.load, df, BROAD_CLASS_NAMES)
        save_json(broad_map_present, BROAD_JSON)
        print("Saved broad classes:", BROAD_JSON)

# ======================
# Driver
# ======================

def run_data_pipeline():
    data_loader = DataProcessing(
        file_path="./datasets/interim/CompleteData_en/dataset_parquet",  # parquet directory or file
        output_path="./datasets/output",
        antibiotics_class_file="./datasets/antibiotic_classification_complete.csv",
        outcome_scheme="ordinal_R2I1S0"  # R=2, I=1, S=0; NaN preserved as <NA>
    )
    run_full_descriptor(
        data_loader.data, name="AMR_2025Q1_COMPLETE_FILTERED",
        out_root="reports/CompleteData_Filtered"
    )

    # If you need to continue with your existing DataLoader:
    from src.controllers.DataLoader import DataLoader
    from src.mappers.top_pathogens import ALL_PATHOGENS

    parquet_dir = "./datasets/structured/dataset_parquet/"
    loader = DataLoader(parquet_dir, pathogen_groups_regex=ALL_PATHOGENS)  # ensure DataLoader reads parquet dirs
    df = loader.get_combined()
    # Needed for just network comparisons and reviewer's replication
    exempt_cols = [
        'NumberOrganisation', 'OrgType', 'Anonymized_Lab', 
        'IsSpecificlyExcluded_Screening', 'IsSpecificlyExcluded_Pathogen', 
        'IsSpecificlyExcluded_PathogenevidenceNegative', 'CSQ', 'CSQMG', 'CSY', 'CSYMG',
        'PathogengroupL1', 'SeasonCode', 'SeasonName', 'PathogenSummary',
        'TypeOrganisation', 'TotalAntibioticsTested'
    ]
    df = df.drop(columns=exempt_cols, errors='ignore')
    
    save_parquet_flat(df=df, out_dir="./datasets/WHO_Aware_data", rows_per_file=250_000)
    save_parquet_flat(df=df, out_dir="./datasets/WHO_Aware_data__", rows_per_file=250_000)
    run_full_descriptor(df, name="AMR_2025Q1_WHO_AWaRe_FILTERING", out_root="reports/CompleteData_WHO_AWaRe_Filtering")
    
    data_loader.generateJson(df)


if __name__ == "__main__":
    run_data_pipeline()
