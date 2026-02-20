from flask import config
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import io # Used to read the string data

# =============================================================================
# 1. CONFIGURE LOGGING
# =============================================================================
# Configure logging to see the cohort creation process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 2. DATACLASS DEFINITION
# =============================================================================

@dataclass
class CohortConfig:
    """Configuration for cohort generation"""
    name: str
    pathogens: Optional[List[str]] = None
    pathogen_genus: Optional[List[str]] = None
    isolate_group: Optional[List[str]] = None  # Filters CSY/CSQ columns
    specimens: Optional[List[str]] = None
    gram_type: Optional[str] = None
    care_type: Optional[str] = None
    ward_type: Optional[str] = None
    care_complexity: Optional[str] = None
    hospital_level: Optional[str] = None
    age_groups: Optional[List[str]] = None
    years: Optional[List[int]] = None
    months: Optional[List[int]] = None
    regions: Optional[List[str]] = None
    sex: Optional[List[str]] = None
    min_sample_size: int = 50
    remove_untested_abx: bool = True
    description: str = ""
    covariates: Optional[List[str]] = None  # Renamed from 'fields' for clarity
    include_default_metadata: bool = True  # Whether to include default metadata columns

# =============================================================================
# 3. THE COHORT GENERATOR "ENGINE"
# =============================================================================

class ProductionCohortGenerator:
    """
    Production-ready cohort generator for antibiotic co-testing analysis
    - Separated pathogen filtering from isolate group (deduplication) filtering.
    - Updated to match new data schema (CareType, AgeGroup, CSY/CSQ).
    """
    
    # Constants for column groups
    METADATA_COLUMNS = [
        'Pathogen', 'GramType', 'Sex', 'Date', 'PathogenGenus',
        'TextMaterialgroupRkiL0', 
        # 'AgeGroup', 'AgeRange', 
        'CareType',
        # 'ARS_HospitalLevelManual', 
        'ARS_WardType', 
        # 'ARS_Region', 
        'Year',
        # 'Month', 'MonthName', 'YearMonth', 'BroadAgeGroup', 'HighLevelAgeRange', 
        # 'Hospital_Priority', 'Care_Complexity', 'Facility_Function',
        # 'CSQ', 'CSQMG', 'CSY', 'CSYMG'
    ]
    
    def __init__(self, df: pd.DataFrame, config_path: Optional[str] = None):
        """
        Initialize cohort generator
        """
        if df.empty:
            raise ValueError("Input dataframe is empty. Cannot initialize generator.")
            
        self.df = df.copy()
        self.available_columns = df.columns.tolist()
        self.cohorts: Dict[str, pd.DataFrame] = {}
        self.cohort_metadata: Dict[str, Dict] = {}
        
        # Setup mappings and validation
        self._validate_data()
        self._setup_mappings()
        self._load_configurations(config_path)
        self.is_pairwise = all(c in self.df.columns for c in ["ab_1","ab_2","a","b","c","d"])

        logger.info(f"CohortGenerator initialized with {len(self.df)} isolates")
    
    def _validate_data(self) -> None:
        if self.df.empty:
            raise ValueError("Input dataframe is empty.")

        self.is_pairwise = all(c in self.df.columns for c in ["ab_1","ab_2","a","b","c","d"])

        if self.is_pairwise:
            # validate counts exist and are numeric
            for c in ["a","b","c","d"]:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0).astype(int)
            return

        # WIDE validation (existing)
        if not any(str(col).endswith("_Tested") for col in self.df.columns):
            raise ValueError("No '_Tested' columns found in DataFrame.")

        tested_columns = [col for col in self.df.columns if str(col).endswith("_Tested")]
        for col in tested_columns:
            col_data = pd.to_numeric(self.df[col], errors="coerce")
            unique_vals = col_data.dropna().unique()
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                logger.warning(f"Column {col} contains non-binary values: {unique_vals}")


    # def _validate_data(self) -> None:
    #     """Validate input data structure and required columns"""
    #     required_meta = [col for col in self.METADATA_COLUMNS if col in self.available_columns]
        
    #     if not any(col.endswith('_Tested') for col in self.available_columns):
    #         raise ValueError("No '_Tested' columns found in DataFrame.")
            
    #     required_columns = required_meta + [
    #         col for col in self.available_columns if col.endswith('_Tested')
    #     ]
        
    #     missing_columns = set(required_columns) - set(self.available_columns)
    #     if missing_columns:
    #         logger.warning(f"Missing optional metadata columns: {missing_columns}")
        
    #     tested_columns = [col for col in self.available_columns if col.endswith('_Tested')]
    #     for col in tested_columns:
    #         col_data = pd.to_numeric(self.df[col], errors='coerce')
    #         unique_vals = col_data.dropna().unique()
    #         if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
    #             logger.warning(f"Column {col} contains non-binary values: {unique_vals}")

    
    def _setup_mappings(self) -> None:
        """Setup clinical mappings and value sets"""
        
        # Map legacy config values to new data values
        self.CARE_TYPE_MAP = {
            'stationär': 'In-Patient',
            'ambulant': 'Out-Patient'
        }
        
        # Available values for categorical fields
        self.available_specimens = self.df['TextMaterialgroupRkiL0'].dropna().unique().tolist()
        self.available_care_types = self.df['CareType'].dropna().unique().tolist()
        self.available_ward_types = self.df['ARS_WardType'].dropna().unique().tolist()
        # self.available_age_groups = self.df['AgeGroup'].dropna().unique().tolist() 
        self.available_gram_types = self.df['GramType'].dropna().unique().tolist()
        # self.available_regions = self.df['ARS_Region'].dropna().unique().tolist()
        self.available_sex = self.df['Sex'].dropna().unique().tolist()
        self.available_care_complexity = self.df['Care_Complexity'].dropna().unique().tolist()
        # self.available_hospital_level = self.df['ARS_HospitalLevelManual'].dropna().unique().tolist()

        logger.debug(f"Available specimens: {self.available_specimens}")
        logger.debug(f"Available care types: {self.available_care_types}")
        # logger.debug(f"Available age groups: {self.available_age_groups}")
    
    def _load_configurations(self, config_path: Optional[str]) -> None:
        """Load predefined cohort configurations"""
        self.predefined_cohorts = {}
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self.predefined_cohorts = config_data.get('predefined_cohorts', {})
                logger.info(f"Loaded {len(self.predefined_cohorts)} predefined cohorts from config")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        
        if not self.predefined_cohorts:
            self._setup_default_predefined_cohorts()
    
    def _setup_default_predefined_cohorts(self) -> None:
        """Setup default clinically relevant cohorts"""
        self.predefined_cohorts = {
            'icu_bloodstream_infections': {
                'isolate_group': ['CSY'], 
                'specimens': ['Blood Culture'],
                'ward_type': 'Intensive Care Unit',
                'min_sample_size': 30,
                'description': 'High-mortality bloodstream infections in ICU (First Isolates per Year)'
            },
            'complicated_uti_elderly': {
                'pathogens': ['Escherichia coli', 'Klebsiella pneumoniae'],
                'specimens': ['Urine'],
                'care_type': 'stationär', 
                'age_groups': ['Elderly'], 
                'description': 'Complicated UTIs in elderly hospitalized patients'
            },
            'community_respiratory': {
                'isolate_group': ['CSYMG'], 
                'specimens': ['Respiratory'], 
                'care_type': 'ambulant', 
                'description': 'Community-acquired respiratory infections (First Isolates per Year/Material)'
            },
            'surgical_site_infections': {
                'pathogens': ['Staphylococcus aureus', 'Escherichia coli'],
                'specimens': ['Wound'],
                'care_type': 'stationär',
                'description': 'Surgical site and wound infections'
            },
            'pediatric_bloodstream': {
                'specimens': ['Blood Culture'],
                'age_groups': ['Pediatric'], 
                'description': 'Bloodstream infections in pediatric population'
            }
        }
    
    def create_cohort(self, config: CohortConfig) -> Optional[pd.DataFrame]:
        """
        Create a cohort based on configuration
        """
        try:
            logger.info(f"Creating cohort: {config.name}")
            
            logger.info(
                f"[{config.name}] config snapshot: "
                f"pathogens={config.pathogens} "
                f"genus={config.pathogen_genus} "
                f"isolate_group={config.isolate_group} "
                f"specimens={config.specimens} "
                f"gram_type={config.gram_type} "
                f"care_type={config.care_type} "
                f"ward_type={config.ward_type} "
                f"care_complexity={config.care_complexity} "
                f"hospital_level={config.hospital_level} "
                f"age_groups={config.age_groups} "
                f"sex={config.sex} "
                f"years={config.years} months={config.months} "
                f"regions={config.regions} "
                f"min_sample_size={config.min_sample_size} "
                f"remove_untested_abx={config.remove_untested_abx}"
            )
            # Start with all data
            # mask = pd.Series([True] * len(self.df))
            mask = pd.Series(True, index=self.df.index)
            self._log_mask_step(config.name, "start", mask)

            mask = self._apply_pathogen_filters(mask, config)
            self._log_mask_step(config.name, "after_pathogen", mask)

            mask = self._apply_isolate_filters(mask, config)
            self._log_mask_step(config.name, "after_isolate", mask)

            mask = self._apply_specimen_filters(mask, config)
            self._log_mask_step(config.name, "after_specimen", mask)

            # mask = self._apply_sex_filters(mask, config)
            # self._log_mask_step(config.name, "after_sex", mask)

            mask = self._apply_demographic_filters(mask, config)
            self._log_mask_step(config.name, "after_demographic", mask)

            mask = self._apply_care_complexity_filters(mask, config)
            self._log_mask_step(config.name, "after_care_complexity", mask)

            # mask = self._apply_hospital_level_filters(mask, config)
            # self._log_mask_step(config.name, "after_hospital_level", mask)

            temporal_mask = self._apply_temporal_filters(mask, config)
            if temporal_mask is None:
                logger.warning(f"[{config.name}] temporal filter returned None")
                return None
            mask = temporal_mask
            self._log_mask_step(config.name, "after_temporal", mask)

            # mask = self._apply_geographic_filters(mask, config)
            # self._log_mask_step(config.name, "after_geographic", mask)

            cohort_df = self.df.loc[mask].copy()
            logger.info(f"[{config.name}] final cohort rows={len(cohort_df):,}")
            if 'Year' in cohort_df.columns:
                year_counts = cohort_df['Year'].value_counts().sort_index()
                logger.info(f"[{config.name}] Year distribution:\n{year_counts}")
            else:
                logger.warning(f"[{config.name}] Year column not present in final cohort")
            # self._log_dedup_distribution(config.name, mask)

            
            # Check sample size
            # if len(cohort_df) < config.min_sample_size:
            #     logger.warning(f"Cohort '{config.name}' too small: {len(cohort_df)} < {config.min_sample_size}")
            #     return None
            if self.is_pairwise:
                # cohort size = total isolates in that stratum (should be constant across pairs)
                N_candidates = (cohort_df["a"] + cohort_df["b"] + cohort_df["c"] + cohort_df["d"]).unique()
                # choose max (or assert 1 unique if pipeline guarantees it)
                N = int(np.max(N_candidates)) if len(N_candidates) else 0
                if N < config.min_sample_size:
                    logger.warning(f"Cohort '{config.name}' too small: N={N} < {config.min_sample_size}")
                    return None
            else:
                if len(cohort_df) < config.min_sample_size:
                    logger.warning(f"Cohort '{config.name}' too small: {len(cohort_df)} < {config.min_sample_size}")
                    return None

            
            # Clean data
            cohort_df = self._clean_cohort_data(cohort_df, config)
            
            # Add metadata
            cohort_df.attrs.update({
                'cohort_name': config.name,
                'sample_size': len(cohort_df),
                'description': config.description,
                'config': config.__dict__,
                'created_at': pd.Timestamp.now()
            })
            
            # Store cohort
            self.cohorts[config.name] = cohort_df
            self._update_cohort_metadata(config.name, cohort_df)
            
            logger.info(f"✓ Successfully created cohort '{config.name}' with {len(cohort_df)} isolates")
            if 'Year' in cohort_df.columns:
                logger.info(f"[{config.name}] Final unique years: {cohort_df['Year'].unique()}")

            return cohort_df
            
        except Exception as e:
            logger.error(f"Error creating cohort '{config.name}': {e}")
            return None
    
    def _apply_care_complexity_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply care complexity filters"""
        if config.care_complexity:
            complexity_values = config.care_complexity if isinstance(config.care_complexity, list) else [config.care_complexity]
            valid_complexities = [c for c in complexity_values if c in self.available_care_complexity]
            if valid_complexities:
                mask &= self.df['Care_Complexity'].isin(valid_complexities)
                logger.debug(f"Filtered by Care_Complexity: {valid_complexities}")
            else:
                logger.warning(f"No valid care complexities found in: {complexity_values}. Available: {self.available_care_complexity}")
        
        return mask
    
    def _apply_hospital_level_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply hospital level filters"""
        if config.hospital_level:
            hospital_level_values = config.hospital_level if isinstance(config.hospital_level, list) else [config.hospital_level]
            valid_levels = [h for h in hospital_level_values if h in self.available_hospital_level]
            if valid_levels:
                mask &= self.df['ARS_HospitalLevelManual'].isin(valid_levels)
                logger.debug(f"Filtered by Hospital Level: {valid_levels}")
            else:
                logger.warning(f"No valid hospital levels found in: {hospital_level_values}. Available: {self.available_hospital_level}")
        
        return mask
    
    def _apply_pathogen_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply pathogen and Gram type filters"""
        
        if config.pathogens:
            mask &= self.df['Pathogen'].isin(config.pathogens)
            logger.debug(f"Filtered by {len(config.pathogens)} specific pathogens")
        
        # Filter by pathogen genus (genus-level) ← NEW
        if config.pathogen_genus:
            mask &= self.df['PathogenGenus'].isin(config.pathogen_genus)
            logger.debug(f"Filtered by {len(config.pathogen_genus)} pathogen genera")
        
    
        if config.gram_type:
            if config.gram_type in self.available_gram_types:
                mask &= self.df['GramType'] == config.gram_type
                logger.debug(f"Filtered by Gram type: {config.gram_type}")
            else:
                logger.warning(f"Unknown Gram type: {config.gram_type}")
        
        return mask

    def _apply_isolate_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply isolate group (deduplication) filters"""

        if not config.isolate_group:
            logger.info(f"[{config.name}] isolate filter: none (config.isolate_group is empty)")
            return mask

        for group_col in config.isolate_group:
            if group_col not in self.df.columns:
                logger.warning(f"[{config.name}] isolate filter: column '{group_col}' not in DataFrame")
                continue

            # count before/after on the CURRENT mask
            before = int(mask.fillna(False).astype(bool).sum())

            col = self.df[group_col]
            # be robust to weird types / missing values
            is_first = (col.astype("string") == "Erstisolat")

            mask = mask & is_first

            after = int(mask.fillna(False).astype(bool).sum())

            # log both the effect and a quick sanity distribution for that col within the cohort
            logger.info(
                f"[{config.name}] isolate filter {group_col}=='Erstisolat': "
                f"{before:,} -> {after:,} (dropped {before-after:,})"
            )

        return mask

    def _log_dedup_distribution(self, cohort_name: str, mask: pd.Series) -> None:
        cols = [c for c in ["CSY", "CSYMG", "CSQ", "CSQMG"] if c in self.df.columns]
        if not cols:
            return

        sub = self.df.loc[mask, cols].astype("string")
        for c in cols:
            vc = sub[c].value_counts(dropna=False).head(5).to_dict()
            logger.info(f"[{cohort_name}] dedup dist within mask: {c} top={vc}")


    def _apply_specimen_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        if not config.specimens:
            logger.info(f"[{config.name}] specimen filter: none (config.specimens is empty)")
            return mask

        specimen_values = config.specimens if isinstance(config.specimens, list) else [config.specimens]
        valid_specimens = [s for s in specimen_values if s in self.available_specimens]

        if not valid_specimens:
            logger.warning(
                f"[{config.name}] specimen filter: no valid specimens in {specimen_values}. "
                f"Available examples={self.available_specimens[:10]}"
            )
            return mask

        before = int(mask.fillna(False).astype(bool).sum())
        mask = mask & self.df['TextMaterialgroupRkiL0'].isin(valid_specimens)
        after = int(mask.fillna(False).astype(bool).sum())
        logger.info(f"[{config.name}] specimen filter {valid_specimens}: {before:,} -> {after:,}")
        return mask

    
    def _apply_sex_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply specimen-related filters"""
        if config.sex:
            sex_values = config.sex if isinstance(config.sex, list) else [config.sex]
            valid_sex = [s for s in sex_values if s in self.available_sex]
            if valid_sex:
                mask &= self.df['Sex'].isin(valid_sex)
                logger.debug(f"Filtered by Sex: {valid_sex}")
            else:
                logger.warning(f"No valid Sex found in: {sex_values}. Available: {self.available_sex}")
        
        return mask
    
    def _apply_demographic_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply demographic filters"""
        
        if config.care_type:
            mapped_care_type = self.CARE_TYPE_MAP.get(config.care_type, config.care_type)
            
            if mapped_care_type in self.available_care_types:
                mask &= self.df['CareType'] == mapped_care_type
                logger.debug(f"Filtered by CareType: {mapped_care_type}")
            else:
                logger.warning(f"Care type '{config.care_type}' (mapped to '{mapped_care_type}') not found. Available: {self.available_care_types}")
        
        if config.ward_type and config.ward_type in self.available_ward_types:
            ward_values = config.ward_type if isinstance(config.ward_type, list) else [config.ward_type]
            mask &= self.df['ARS_WardType'].isin(ward_values) if isinstance(ward_values, list) else (self.df['ARS_WardType'] == ward_values)
            logger.debug(f"Filtered by WardType: {ward_values}")

        # if config.age_groups:
        #     age_group_values = config.age_groups if isinstance(config.age_groups, list) else [config.age_groups]
        #     valid_age_groups = [ag for ag in age_group_values if ag in self.available_age_groups]
        #     if valid_age_groups:
        #         mask &= self.df['AgeGroup'].isin(valid_age_groups)
        #         logger.debug(f"Filtered by AgeGroups: {valid_age_groups}")
        #     else:
        #         logger.warning(f"No valid age groups found in: {age_group_values}. Available: {self.available_age_groups}")
        
        return mask
    
    def _log_mask_step(self, cohort_name: str, step: str, mask: pd.Series) -> None:
        try:
            same_index = mask.index.equals(self.df.index)
            # handle dtype quirks
            n_true = int(mask.fillna(False).astype(bool).sum())
            logger.info(
                f"[{cohort_name}] {step}: kept={n_true:,}/{len(self.df):,} "
                f"({(n_true/len(self.df)) if len(self.df) else 0:.2%}) "
                f"index_ok={same_index} mask_index={type(mask.index).__name__} df_index={type(self.df.index).__name__}"
            )
            if not same_index:
                logger.warning(f"[{cohort_name}] INDEX MISMATCH at step '{step}'")
        except Exception as e:
            logger.warning(f"[{cohort_name}] failed logging step '{step}': {e}")
            

    def _apply_temporal_filters(self, mask: pd.Series, config: CohortConfig) -> Optional[pd.Series]:
        """
        Apply temporal filters (Years, Months) – use string comparison for years.
        """
        # ---- Years ----
        if config.years is not None and len(config.years) > 0:
            if 'Year' not in self.df.columns:
                logger.warning(f"[{config.name}] Year column not present, cannot apply year filter.")
                return None

            # Use the column as strings (no numeric conversion)
            year_col = self.df['Year'].astype(str)
            wanted_years = [str(y) for y in config.years]  # ensure strings
            available_years_set = set(year_col.dropna().unique())
            available_years_sorted = sorted(available_years_set)

            effective_years = [y for y in wanted_years if y in available_years_set]

            logger.info(
                f"[{config.name}] temporal: wanted_years={wanted_years} "
                f"effective_years={effective_years} "
                f"available_years={available_years_sorted}"
            )

            if not effective_years:
                logger.warning(f"[{config.name}] None of the requested years {wanted_years} are present in the data.")
                return None

            before = int(mask.fillna(False).astype(bool).sum())
            mask = mask & year_col.isin(effective_years)
            after = int(mask.fillna(False).astype(bool).sum())
            logger.info(f"[{config.name}] temporal year filter: {before:,} -> {after:,}")

        # ---- Months ----
        if config.months is not None and len(config.months) > 0:
            if 'Month' not in self.df.columns:
                logger.warning(f"[{config.name}] Month column not present, cannot apply month filter.")
                return None
            # For months, you can keep numeric if you prefer, but ensure consistency
            month_col = pd.to_numeric(self.df['Month'], errors='coerce').astype('Int64')
            wanted_months = [int(m) for m in config.months]  # assume config.months are strings/ints
            before = int(mask.fillna(False).astype(bool).sum())
            mask = mask & month_col.isin(wanted_months)
            after = int(mask.fillna(False).astype(bool).sum())
            logger.info(f"[{config.name}] temporal month filter: {before:,} -> {after:,}")

        return mask

    # def _apply_temporal_filters(self, mask: pd.Series, config: CohortConfig) -> Optional[pd.Series]:
    #     """Apply temporal filters (Years, Months)"""

    #     if 'Year' not in self.df.columns or 'Month' not in self.df.columns:
    #         logger.warning("Year or Month column not present, skipping temporal filters.")
    #         return mask

    #     year_col  = pd.to_numeric(self.df['Year'],  errors='coerce').astype('Int64')
    #     month_col = pd.to_numeric(self.df['Month'], errors='coerce').astype('Int64')

    #     # ---- Years ----
    #     if config.years is not None and len(config.years) > 0:
    #         wanted_years = list(dict.fromkeys(config.years))
    #         available_years_set = set(year_col.dropna().unique().tolist())
    #         available_years_sorted = sorted(available_years_set)

    #         effective_years = [y for y in wanted_years if y in available_years_set]

    #         logger.info(
    #             f"[{config.name}] temporal: wanted_years={wanted_years} "
    #             f"effective_years={effective_years} "
    #             f"available_years={available_years_sorted}"
    #         )

    #         if not effective_years:
    #             logger.warning(f"[{config.name}] None of the requested years {wanted_years} are present in the data.")
    #             return None

    #         before = int(mask.fillna(False).astype(bool).sum())
    #         mask = mask & year_col.isin(effective_years)
    #         after = int(mask.fillna(False).astype(bool).sum())
    #         logger.info(f"[{config.name}] temporal year filter: {before:,} -> {after:,}")
    #         # NEW: Log the unique years present in the filtered data
    #         filtered_years = self.df.loc[mask, 'Year'].dropna().unique()
    #         logger.info(f"[{config.name}] Years present after filter: {sorted(filtered_years)}")
    
    #     # ---- Months ----
    #     if config.months is not None and len(config.months) > 0:
    #         wanted_months = list(dict.fromkeys(config.months))
    #         before = int(mask.fillna(False).astype(bool).sum())
    #         mask = mask & month_col.isin(wanted_months)
    #         after = int(mask.fillna(False).astype(bool).sum())
    #         logger.info(f"[{config.name}] temporal month filter: {before:,} -> {after:,}")

    #     return mask


    # def _apply_geographic_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
    #     """Apply geographic filters"""
    #     if config.regions:
    #         valid_regions = [r for r in config.regions if r in self.available_regions]
    #         if valid_regions:
    #             mask &= self.df['ARS_Region'].isin(valid_regions)
    #             logger.debug(f"Filtered by Regions: {valid_regions}")
    #         else:
    #             logger.warning(f"No valid regions found in: {config.regions}. Available: {self.available_regions}")
        
    #     return mask    
    
    
    def _clean_cohort_data(self, cohort_df: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
        available_columns = cohort_df.columns.tolist()
        default_metadata_cols = [col for col in self.METADATA_COLUMNS if col in available_columns]

        # choose covariates
        if config.covariates:
            valid_covariates = [cov for cov in config.covariates if cov in available_columns]
            non_antibiotic_cols = (default_metadata_cols + valid_covariates) if config.include_default_metadata else valid_covariates
            non_antibiotic_cols = list(dict.fromkeys(non_antibiotic_cols))
        else:
            non_antibiotic_cols = default_metadata_cols

        if self.is_pairwise:
            core = [c for c in ["ab_1","ab_2","a","b","c","d"] if c in available_columns]
            final_columns = list(dict.fromkeys(non_antibiotic_cols + core))
            out = cohort_df[final_columns].copy()

            # remove untested pairs if requested (interpretation: no one tested either → meaningless)
            if config.remove_untested_abx:
                # require at least one unilateral or joint test exists
                out = out[(out["a"] + out["b"] + out["c"]) > 0]

            return out

        # ---- WIDE original behavior ----
        antibiotic_columns = [col for col in available_columns if str(col).endswith("_Tested")]
        all_columns = list(dict.fromkeys(non_antibiotic_cols + antibiotic_columns))

        if config.remove_untested_abx and antibiotic_columns:
            numeric_abx_df = cohort_df[antibiotic_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
            tested_abx_mask = numeric_abx_df.sum() > 0
            tested_antibiotic_columns = numeric_abx_df.columns[tested_abx_mask].tolist()
            all_columns = [c for c in all_columns if (not str(c).endswith("_Tested")) or (c in tested_antibiotic_columns)]

        final_columns = [col for col in all_columns if col in available_columns]
        return cohort_df[final_columns].copy()



    # def _clean_cohort_data(self, cohort_df: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
    #     """Clean and prepare cohort data - ALWAYS include all antibiotics, select other covariates"""
        
    #     # Start with all available columns in the filtered cohort
    #     available_columns = cohort_df.columns.tolist()
        
    #     # IDENTIFY COLUMN GROUPS
    #     # 1. Always include all antibiotic tested columns
    #     antibiotic_columns = [col for col in available_columns if col.endswith('_Tested')]
        
    #     # 2. Handle covariate selection
    #     default_metadata_cols = [col for col in self.METADATA_COLUMNS if col in available_columns]
        
    #     if config.covariates:
    #         # Validate requested covariates
    #         valid_covariates = [cov for cov in config.covariates if cov in available_columns]
    #         missing_covariates = set(config.covariates) - set(valid_covariates)
            
    #         if missing_covariates:
    #             logger.warning(f"Covariates not found in data: {missing_covariates}")
            
    #         # Combine covariates with default metadata if requested
    #         if config.include_default_metadata:
    #             non_antibiotic_cols = list(dict.fromkeys(default_metadata_cols + valid_covariates))
    #         else:
    #             non_antibiotic_cols = valid_covariates
    #     else:
    #         # Use only default metadata if no covariates specified
    #         non_antibiotic_cols = default_metadata_cols
        
    #     # COMBINE ALL SELECTED COLUMNS: Antibiotics + Non-antibiotic columns
    #     all_columns = list(dict.fromkeys(non_antibiotic_cols + antibiotic_columns))
        
    #     # Handle removal of untested antibiotics if requested
    #     if config.remove_untested_abx and antibiotic_columns:
    #         # Convert antibiotic columns to numeric and check which have any testing
    #         numeric_abx_df = cohort_df[antibiotic_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    #         tested_abx_mask = numeric_abx_df.sum() > 0
    #         tested_antibiotic_columns = numeric_abx_df.columns[tested_abx_mask].tolist()
    #         untested_antibiotic_columns = set(antibiotic_columns) - set(tested_antibiotic_columns)
            
    #         # Remove untested antibiotics from our final column selection
    #         all_columns = [col for col in all_columns if col not in untested_antibiotic_columns]
            
    #         logger.debug(f"Removed {len(untested_antibiotic_columns)} untested antibiotics, kept {len(tested_antibiotic_columns)} tested antibiotics")
        
    #     # Ensure we only keep columns that exist in the dataframe
    #     final_columns = [col for col in all_columns if col in available_columns]
        
    #     # Apply column selection
    #     cohort_df = cohort_df[final_columns]
        
    #     # Log the composition of the final cohort
    #     final_antibiotic_cols = [col for col in final_columns if col.endswith('_Tested')]
    #     final_covariate_cols = [col for col in final_columns if not col.endswith('_Tested')]
        
    #     logger.debug(f"Final cohort '{config.name}': {len(cohort_df)} rows, {len(final_columns)} columns")
    #     logger.debug(f"  - Antibiotics: {len(final_antibiotic_cols)} columns")
    #     logger.debug(f"  - Covariates: {len(final_covariate_cols)} columns")
        
    #     return cohort_df
    
    
    
    def get_available_covariates(self) -> Dict[str, List[str]]:
        """Get available covariates (non-antibiotic columns) categorized by type"""
        antibiotic_columns = [col for col in self.available_columns if col.endswith('_Tested')]
        non_antibiotic_columns = [col for col in self.available_columns if col not in antibiotic_columns]
        
        categories = {
            'demographic': [col for col in non_antibiotic_columns if col in ['Sex', 'AgeGroup', 'AgeRange', 'BroadAgeGroup', 'HighLevelAgeRange']],
            'clinical': [col for col in non_antibiotic_columns if col in ['CareType', 'ARS_WardType', 'TextMaterialgroupRkiL0', 'Hospital_Priority', 'Care_Complexity', 'Facility_Function']],
            'temporal': [col for col in non_antibiotic_columns if col in ['Year', 'Month', 'MonthName', 'YearMonth', 'Date']],
            'geographic': [col for col in non_antibiotic_columns if col in ['ARS_Region', 'ARS_HospitalLevelManual']],
            'pathogen': [col for col in non_antibiotic_columns if col in ['Pathogen', 'PathogenGenus', 'GramType']],
            'deduplication': [col for col in non_antibiotic_columns if col in ['CSQ', 'CSQMG', 'CSY', 'CSYMG']],
            'all_metadata': [col for col in non_antibiotic_columns if col in self.METADATA_COLUMNS]
        }
        
        # Filter to only include existing columns and remove empty categories
        return {category: cols for category, cols in categories.items() if cols}

    # def get_antibiotic_list(self) -> List[str]:
    #     """Get list of all available antibiotic columns"""
    #     return [col for col in self.available_columns if col.endswith('_Tested')]
    
    def get_antibiotic_list(self) -> List[str]:
        if self.is_pairwise:
            return sorted(set(self.df["ab_1"].astype(str)).union(set(self.df["ab_2"].astype(str))))
        return [c for c in self.available_columns if str(c).endswith("_Tested")]


    def validate_covariates(self, covariates: List[str]) -> Tuple[List[str], List[str]]:
        """Validate requested covariates against available non-antibiotic columns"""
        antibiotic_columns = [col for col in self.available_columns if col.endswith('_Tested')]
        non_antibiotic_columns = [col for col in self.available_columns if col not in antibiotic_columns]
        
        valid_covariates = [cov for cov in covariates if cov in non_antibiotic_columns]
        invalid_covariates = [cov for cov in covariates if cov not in non_antibiotic_columns]
        
        # Also check if any requested covariates are actually antibiotics
        antibiotic_covariates = [cov for cov in covariates if cov in antibiotic_columns]
        if antibiotic_covariates:
            logger.info(f"Note: These are antibiotic columns (always included): {antibiotic_covariates}")
        
        return valid_covariates, invalid_covariates
        
    
    
    def _update_cohort_metadata(self, cohort_name: str, cohort_df: pd.DataFrame) -> None:
        """Update cohort metadata storage"""
        meta = {
            'sample_size': len(cohort_df),
            'antibiotics_tested': len([col for col in cohort_df.columns if col.endswith('_Tested')]),
        }
        
        for col, key in [
            ('Pathogen', 'pathogens'), 
            ('TextMaterialgroupRkiL0', 'specimens'),
            ('CareType', 'care_types'),
            ('ARS_WardType', 'ward_types'),
            ('Year', 'years')
        ]:
            if col in cohort_df.columns:
                meta[key] = cohort_df[col].value_counts().to_dict()
            else:
                meta[key] = {}
                
        self.cohort_metadata[cohort_name] = meta
    
    def create_cohort_from_dict(self, config_dict: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Create cohort from dictionary configuration"""
        try:
            config = CohortConfig(**config_dict)
            return self.create_cohort(config)
        except Exception as e:
            logger.error(f"Error creating cohort from dict: {e}")
            return None
    
    def create_predefined_cohort(self, cohort_key: str) -> Optional[pd.DataFrame]:
        """Create a predefined cohort"""
        if cohort_key not in self.predefined_cohorts:
            logger.error(f"Predefined cohort not found: {cohort_key}")
            return None
        
        config_dict = self.predefined_cohorts[cohort_key].copy()
        config_dict['name'] = cohort_key
        
        return self.create_cohort_from_dict(config_dict)
    
    def create_all_predefined_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Create all predefined cohorts"""
        successful_cohorts = {}
        
        for cohort_key in self.predefined_cohorts:
            cohort_df = self.create_predefined_cohort(cohort_key)
            if cohort_df is not None:
                successful_cohorts[cohort_key] = cohort_df
        
        logger.info(f"Created {len(successful_cohorts)} out of {len(self.predefined_cohorts)} predefined cohorts")
        return successful_cohorts
    
    def create_comparison_cohorts(self, base_config: CohortConfig, 
                               compare_by: str, values: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Create multiple cohorts for comparison
        """
        comparison_cohorts = {}
        
        list_fields = ['pathogens', 'isolate_group', 'specimens', 'age_groups', 'years', 'months', 'regions', 'sex']
        
        for value in values:
            compare_config_dict = base_config.__dict__.copy()
            
            if compare_by in list_fields:
                compare_config_dict[compare_by] = [value]
            else:
                compare_config_dict[compare_by] = value
            
            safe_value = str(value).replace(' ', '_').replace('<', 'lt').replace('>', 'gt').replace('/', '_')
            compare_config_dict['name'] = f"{base_config.name}_{safe_value}"
            
            cohort_df = self.create_cohort_from_dict(compare_config_dict)
            if cohort_df is not None:
                comparison_cohorts[compare_config_dict['name']] = cohort_df
        
        return comparison_cohorts
    
    def get_cohort_summary(self) -> pd.DataFrame:
        """Get summary of all created cohorts"""
        summary_data = []
        
        for name, metadata in self.cohort_metadata.items():
            summary_data.append({
                'cohort_name': name,
                'sample_size': metadata.get('sample_size', 0),
                'antibiotics_tested': metadata.get('antibiotics_tested', 0),
                'unique_pathogens': len(metadata.get('pathogens', {})),
                'most_common_pathogen': max(metadata.get('pathogens', {}).items(), key=lambda x: x[1])[0] if metadata.get('pathogens') else 'None',
                'specimen_distribution': str(metadata.get('specimens', {})),
                'care_type_distribution': str(metadata.get('care_types', {}))
            })
        
        return pd.DataFrame(summary_data)
    
    def analyze_cohort_completeness(self, cohort_name: str) -> pd.DataFrame:
        """Analyze testing completeness for a cohort"""
        if cohort_name not in self.cohorts:
            logger.error(f"Cohort not found: {cohort_name}")
            return pd.DataFrame()
        
        cohort_df = self.cohorts[cohort_name]
        abx_columns = [col for col in cohort_df.columns if col.endswith('_Tested')]
        
        if not abx_columns:
            logger.warning(f"No '_Tested' columns found in cohort {cohort_name}.")
            return pd.DataFrame()
            
        numeric_abx_df = cohort_df[abx_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        completeness = numeric_abx_df.mean().sort_values(ascending=False)
        completeness_df = pd.DataFrame({
            'antibiotic': completeness.index,
            'testing_rate': completeness.values,
            'tested_count': numeric_abx_df.sum().loc[completeness.index]
        })
        
        logger.info(f"Completeness analysis for {cohort_name}:")
        logger.info(f"  Total isolates: {len(cohort_df)}")
        logger.info(f"  Antibiotics tested: {len(abx_columns)}")
        logger.info(f"  Mean testing rate: {completeness.mean():.1%}")
        
        return completeness_df
    
    def save_cohort(self, cohort_name: str, filepath: str) -> bool:
        """Save cohort to file"""
        if cohort_name not in self.cohorts:
            logger.error(f"Cohort not found: {cohort_name}")
            return False
        
        try:
            self.cohorts[cohort_name].to_csv(filepath, index=False)
            logger.info(f"Saved cohort '{cohort_name}' to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving cohort: {e}")
            return False
    
    def save_configuration(self, filepath: str) -> bool:
        """Save current configuration to file"""
        try:
            config_data = {
                'predefined_cohorts': self.predefined_cohorts,
                'created_cohorts': list(self.cohorts.keys()),
                'metadata': self.cohort_metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Saved configuration to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False