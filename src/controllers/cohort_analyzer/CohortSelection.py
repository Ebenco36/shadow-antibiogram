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
        'Pathogen', 'PathogengroupL1', 'GramType', 'Sex', 'Date', 'PathogenGenus',
        'TextMaterialgroupRkiL0', 'AgeGroup', 'AgeRange', 'CareType',
        'ARS_HospitalLevelManual', 'ARS_WardType', 'ARS_Region', 'Year',
        'Month', 'MonthName', 'YearMonth', 'SeasonCode', 'SeasonName',
        'PathogenSummary', 'BroadAgeGroup', 'HighLevelAgeRange', 
        'Hospital_Priority', 'Care_Complexity', 'Facility_Function',
        'CSQ', 'CSQMG', 'CSY', 'CSYMG'
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
        
        logger.info(f"CohortGenerator initialized with {len(self.df)} isolates")
    
    def _validate_data(self) -> None:
        """Validate input data structure and required columns"""
        required_meta = [col for col in self.METADATA_COLUMNS if col in self.available_columns]
        
        if not any(col.endswith('_Tested') for col in self.available_columns):
            raise ValueError("No '_Tested' columns found in DataFrame.")
            
        required_columns = required_meta + [
            col for col in self.available_columns if col.endswith('_Tested')
        ]
        
        missing_columns = set(required_columns) - set(self.available_columns)
        if missing_columns:
            logger.warning(f"Missing optional metadata columns: {missing_columns}")
        
        tested_columns = [col for col in self.available_columns if col.endswith('_Tested')]
        for col in tested_columns:
            col_data = pd.to_numeric(self.df[col], errors='coerce')
            unique_vals = col_data.dropna().unique()
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                logger.warning(f"Column {col} contains non-binary values: {unique_vals}")

    
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
        self.available_age_groups = self.df['AgeGroup'].dropna().unique().tolist() 
        self.available_gram_types = self.df['GramType'].dropna().unique().tolist()
        self.available_regions = self.df['ARS_Region'].dropna().unique().tolist()
        self.available_sex = self.df['Sex'].dropna().unique().tolist()
        self.available_care_complexity = self.df['Care_Complexity'].dropna().unique().tolist()
        self.available_hospital_level = self.df['ARS_HospitalLevelManual'].dropna().unique().tolist()

        logger.debug(f"Available specimens: {self.available_specimens}")
        logger.debug(f"Available care types: {self.available_care_types}")
        logger.debug(f"Available age groups: {self.available_age_groups}")
    
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
            
            # Start with all data
            mask = pd.Series([True] * len(self.df))
            
            # Apply filters in logical order
            mask = self._apply_pathogen_filters(mask, config)  # Filter by pathogen name/gram type
            mask = self._apply_isolate_filters(mask, config)   # Filter by deduplication status
            mask = self._apply_specimen_filters(mask, config)  # Filter by specimen
            mask = self._apply_sex_filters(mask, config)  # Filter by Sex
            mask = self._apply_demographic_filters(mask, config) # Filter by patient demo
            mask = self._apply_care_complexity_filters(mask, config)
            mask = self._apply_hospital_level_filters(mask, config)

            temporal_mask = self._apply_temporal_filters(mask, config)
            if temporal_mask is None:
                logger.warning(f"Cohort '{config.name}' returned no data due to temporal filter (e.g., requested years not found).")
                return None
            mask = temporal_mask
            
            mask = self._apply_geographic_filters(mask, config)

            # Apply final mask
            cohort_df = self.df[mask].copy()
            
            # Check sample size
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
        
        if config.isolate_group:
            for group_col in config.isolate_group:
                if group_col in self.df.columns:
                    mask &= (self.df[group_col] == 'Erstisolat')
                    logger.debug(f"Filtered by isolate group: {group_col} == 'Erstisolat'")
                else:
                    logger.warning(f"Isolate group column '{group_col}' not in DataFrame.")
        
        return mask
    
    def _apply_specimen_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply specimen-related filters"""
        if config.specimens:
            specimen_values = config.specimens if isinstance(config.specimens, list) else [config.specimens]
            valid_specimens = [s for s in specimen_values if s in self.available_specimens]
            if valid_specimens:
                mask &= self.df['TextMaterialgroupRkiL0'].isin(valid_specimens)
                logger.debug(f"Filtered by specimens: {valid_specimens}")
            else:
                logger.warning(f"No valid specimens found in: {specimen_values}. Available: {self.available_specimens}")
        
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

        if config.age_groups:
            age_group_values = config.age_groups if isinstance(config.age_groups, list) else [config.age_groups]
            valid_age_groups = [ag for ag in age_group_values if ag in self.available_age_groups]
            if valid_age_groups:
                mask &= self.df['AgeGroup'].isin(valid_age_groups)
                logger.debug(f"Filtered by AgeGroups: {valid_age_groups}")
            else:
                logger.warning(f"No valid age groups found in: {age_group_values}. Available: {self.available_age_groups}")
        
        return mask
    
    def _apply_temporal_filters(self, mask: pd.Series, config: CohortConfig) -> Optional[pd.Series]:
        """Apply temporal filters (Years, Months)"""
        
        if 'Year' not in self.df.columns or 'Month' not in self.df.columns:
            logger.warning("Year or Month column not present, skipping temporal filters.")
            return mask
            
        year_col  = pd.to_numeric(self.df['Year'],  errors='coerce').astype('Int64')
        month_col = pd.to_numeric(self.df['Month'], errors='coerce').astype('Int64')

        if config.years is not None and len(config.years) > 0:
            wanted_years = list(dict.fromkeys(config.years))
            available_years_set = set(year_col.dropna().unique().tolist())
            
            effective_years = [y for y in wanted_years if y in available_years_set]
            
            if not effective_years:
                logger.warning(f"None of the requested years {wanted_years} are present in the data.")
                return None 

            mask = mask & year_col.isin(effective_years)
            logger.debug(f"Filtered by Years: {effective_years}")

        if config.months is not None and len(config.months) > 0:
            wanted_months = list(dict.fromkeys(config.months))
            mask = mask & month_col.isin(wanted_months)
            logger.debug(f"Filtered by Months: {wanted_months}")

        return mask

    def _apply_geographic_filters(self, mask: pd.Series, config: CohortConfig) -> pd.Series:
        """Apply geographic filters"""
        if config.regions:
            valid_regions = [r for r in config.regions if r in self.available_regions]
            if valid_regions:
                mask &= self.df['ARS_Region'].isin(valid_regions)
                logger.debug(f"Filtered by Regions: {valid_regions}")
            else:
                logger.warning(f"No valid regions found in: {config.regions}. Available: {self.available_regions}")
        
        return mask
    
    # def _clean_cohort_data(self, cohort_df: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
    #     """Clean and prepare cohort data"""
    #     if config.remove_untested_abx:
    #         abx_columns = [col for col in cohort_df.columns if col.endswith('_Tested')]
    #         numeric_abx_df = cohort_df[abx_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
            
    #         tested_abx_mask = numeric_abx_df.sum() > 0
    #         tested_abx = numeric_abx_df.columns[tested_abx_mask].tolist()
            
    #         metadata_cols = [col for col in self.METADATA_COLUMNS if col in cohort_df.columns]
    #         all_columns = metadata_cols + tested_abx
    #         all_columns = list(dict.fromkeys(all_columns))
            
    #         cohort_df = cohort_df[all_columns]
    #         logger.debug(f"Removed untested antibiotics, kept {len(tested_abx)} tested antibiotics")
        
    #     return cohort_df
    
    
    
    def _clean_cohort_data(self, cohort_df: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
        """Clean and prepare cohort data - ALWAYS include all antibiotics, select other covariates"""
        
        # Start with all available columns in the filtered cohort
        available_columns = cohort_df.columns.tolist()
        
        # IDENTIFY COLUMN GROUPS
        # 1. Always include all antibiotic tested columns
        antibiotic_columns = [col for col in available_columns if col.endswith('_Tested')]
        
        # 2. Handle covariate selection
        default_metadata_cols = [col for col in self.METADATA_COLUMNS if col in available_columns]
        
        if config.covariates:
            # Validate requested covariates
            valid_covariates = [cov for cov in config.covariates if cov in available_columns]
            missing_covariates = set(config.covariates) - set(valid_covariates)
            
            if missing_covariates:
                logger.warning(f"Covariates not found in data: {missing_covariates}")
            
            # Combine covariates with default metadata if requested
            if config.include_default_metadata:
                non_antibiotic_cols = list(dict.fromkeys(default_metadata_cols + valid_covariates))
            else:
                non_antibiotic_cols = valid_covariates
        else:
            # Use only default metadata if no covariates specified
            non_antibiotic_cols = default_metadata_cols
        
        # COMBINE ALL SELECTED COLUMNS: Antibiotics + Non-antibiotic columns
        all_columns = list(dict.fromkeys(non_antibiotic_cols + antibiotic_columns))
        
        # Handle removal of untested antibiotics if requested
        if config.remove_untested_abx and antibiotic_columns:
            # Convert antibiotic columns to numeric and check which have any testing
            numeric_abx_df = cohort_df[antibiotic_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
            tested_abx_mask = numeric_abx_df.sum() > 0
            tested_antibiotic_columns = numeric_abx_df.columns[tested_abx_mask].tolist()
            untested_antibiotic_columns = set(antibiotic_columns) - set(tested_antibiotic_columns)
            
            # Remove untested antibiotics from our final column selection
            all_columns = [col for col in all_columns if col not in untested_antibiotic_columns]
            
            logger.debug(f"Removed {len(untested_antibiotic_columns)} untested antibiotics, kept {len(tested_antibiotic_columns)} tested antibiotics")
        
        # Ensure we only keep columns that exist in the dataframe
        final_columns = [col for col in all_columns if col in available_columns]
        
        # Apply column selection
        cohort_df = cohort_df[final_columns]
        
        # Log the composition of the final cohort
        final_antibiotic_cols = [col for col in final_columns if col.endswith('_Tested')]
        final_covariate_cols = [col for col in final_columns if not col.endswith('_Tested')]
        
        logger.debug(f"Final cohort '{config.name}': {len(cohort_df)} rows, {len(final_columns)} columns")
        logger.debug(f"  - Antibiotics: {len(final_antibiotic_cols)} columns")
        logger.debug(f"  - Covariates: {len(final_covariate_cols)} columns")
        
        return cohort_df
    
    
    
    def get_available_covariates(self) -> Dict[str, List[str]]:
        """Get available covariates (non-antibiotic columns) categorized by type"""
        antibiotic_columns = [col for col in self.available_columns if col.endswith('_Tested')]
        non_antibiotic_columns = [col for col in self.available_columns if col not in antibiotic_columns]
        
        categories = {
            'demographic': [col for col in non_antibiotic_columns if col in ['Sex', 'AgeGroup', 'AgeRange', 'BroadAgeGroup', 'HighLevelAgeRange']],
            'clinical': [col for col in non_antibiotic_columns if col in ['CareType', 'ARS_WardType', 'TextMaterialgroupRkiL0', 'Hospital_Priority', 'Care_Complexity', 'Facility_Function']],
            'temporal': [col for col in non_antibiotic_columns if col in ['Year', 'Month', 'MonthName', 'YearMonth', 'SeasonCode', 'SeasonName', 'Date']],
            'geographic': [col for col in non_antibiotic_columns if col in ['ARS_Region', 'ARS_HospitalLevelManual']],
            'pathogen': [col for col in non_antibiotic_columns if col in ['Pathogen', 'PathogengroupL1', 'PathogenGenus', 'GramType', 'PathogenSummary']],
            'deduplication': [col for col in non_antibiotic_columns if col in ['CSQ', 'CSQMG', 'CSY', 'CSYMG']],
            'all_metadata': [col for col in non_antibiotic_columns if col in self.METADATA_COLUMNS]
        }
        
        # Filter to only include existing columns and remove empty categories
        return {category: cols for category, cols in categories.items() if cols}

    def get_antibiotic_list(self) -> List[str]:
        """Get list of all available antibiotic columns"""
        return [col for col in self.available_columns if col.endswith('_Tested')]

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






# =============================================================================
# 3. USE-CASE DRIVEN COHORT SELECTOR
# =============================================================================

class UseCaseDrivenCohortSelector(ProductionCohortGenerator):
    """
    Creates cohorts specifically designed for the four key use cases
    """
    
    def create_use_case_cohorts(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Creates all cohorts for the four main use cases"""
        use_case_cohorts = {}
        
        use_case_cohorts['diagnostic_cascade_optimization'] = self._create_cascade_cohorts()
        use_case_cohorts['testing_pattern_evolution'] = self._create_evolution_cohorts()
        use_case_cohorts['diagnostic_stewardship'] = self._create_stewardship_cohorts()
        use_case_cohorts['context_aware_protocols'] = self._create_context_cohorts()
        
        logging.info(f"Created cohorts for {len(use_case_cohorts)} use cases")
        return use_case_cohorts
    
    def _create_cascade_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Use Case 1: Diagnostic Cascade Optimization"""
        cascade_cohorts = {}
        
        cascade_configs = [
            CohortConfig(
                name='sepsis_cascade_icu',
                pathogens=['Escherichia coli', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa', 'Staphylococcus aureus'],
                specimens=['Blood Culture'],
                ward_type='Intensive Care Unit',
                care_type='In-Patient',
                isolate_group=['CSYMG'],
                min_sample_size=30,
                description='Optimal testing sequences for sepsis in critical care settings'
            ),
            CohortConfig(
                name='complicated_uti_cascade',
                pathogens=['Escherichia coli', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa', 'Proteus mirabilis'],
                specimens=['Urine'],
                care_type='In-Patient',
                age_groups=['Elderly (70+ years)'],
                isolate_group=['CSYMG'],
                min_sample_size=40,
                description='Testing sequences for complicated urinary tract infections'
            )
        ]
        
        for config in cascade_configs:
            cohort = self.create_cohort(config)
            if cohort is not None:
                cascade_cohorts[config.name] = cohort
        
        return cascade_cohorts
    
    def _create_evolution_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Use Case 2: Testing Pattern Evolution - Individual Years"""
        evolution_cohorts = {}
        
        available_years = sorted(self.df['Year'].dropna().unique())
        
        # Create individual cohorts for each year with sufficient data
        for year in available_years:
            year_int = int(year)
            
            # Check if this year has enough data
            year_data = self.df[self.df['Year'] == year]
            ecoli_uti_data = year_data[
                (year_data['Pathogen'] == 'Escherichia coli') &
                (year_data['TextMaterialgroupRkiL0'] == 'Urine') &
                (year_data['CareType'] == 'In-Patient') &
                (year_data['CSY'] == 'Erstisolat')
            ]
            
            if len(ecoli_uti_data) >= 20:  # Only create if minimum sample size
                period_config = CohortConfig(
                    name=f"evolution_ecoli_uti_{year_int}",
                    pathogens=['Escherichia coli'],
                    specimens=['Urine'], 
                    care_type='In-Patient',
                    isolate_group=['CSY'],
                    years=[year_int],
                    min_sample_size=20,
                    description=f"EColi UTI testing patterns in {year_int}"
                )
                cohort = self.create_cohort(period_config)
                if cohort is not None:
                    evolution_cohorts[period_config.name] = cohort
        
        return evolution_cohorts
    
    def _create_stewardship_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Use Case 3: Diagnostic Stewardship & Efficiency Analysis"""
        stewardship_cohorts = {}
        
        stewardship_configs = [
            CohortConfig(
                name="stewardship_icu_vs_ward",
                specimens=['Blood Culture'],
                isolate_group=['CSY'],
                years=[2022, 2023] if 2022 in self.df['Year'].unique() else None,
                ward_type=["Normal Ward", "Intensive Care Unit"],
                min_sample_size=20,
                description="Compare Watch/Reserve antibiotic testing in ICU vs general ward"
            )
        ]
        
        for config in stewardship_configs:
            cohort = self.create_cohort(config)
            if cohort is not None:
                stewardship_cohorts[config.name] = cohort
        
        return stewardship_cohorts
    
    def _create_context_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Use Case 4: Context-Aware Diagnostic Protocols"""
        context_cohorts = {}
        
        base_contexts = [
            {
                'name': 'context_ecoli_blood',
                'pathogens': ['Escherichia coli'],
                'specimens': ['Blood Culture'],
                'ward_type': ["Normal Ward", "Intensive Care Unit"],
                'description': 'EColi in blood - expect broad testing panel'
            },
            {
                'name': 'context_ecoli_urine', 
                'pathogens': ['Escherichia coli'],
                'specimens': ['Urine'],
                'ward_type': ["Normal Ward", "Intensive Care Unit"],
                'description': 'EColi in urine - expect narrow testing panel'
            }
        ]
        
        for ctx in base_contexts:
            config = CohortConfig(
                name=ctx['name'],
                pathogens=ctx['pathogens'],
                specimens=ctx['specimens'],
                isolate_group=['CSYMG'],
                min_sample_size=25,
                description=ctx['description']
            )
            cohort = self.create_cohort(config)
            if cohort is not None:
                context_cohorts[config.name] = cohort
        
        return context_cohorts

# =============================================================================
# 4. NETWORK COMPARISON STRATEGY
# =============================================================================

class NetworkComparisonStrategy(UseCaseDrivenCohortSelector):
    """
    Handles splitting cohorts by metadata for comparative network analysis
    """
    
    def split_cohort_by_metadata(self, cohort_name: str, split_column: str, 
                               values: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Split an existing cohort by a metadata column to create comparison groups
        """
        if cohort_name not in self.cohorts:
            logging.error(f"Cohort '{cohort_name}' not found")
            return {}
        
        cohort_df = self.cohorts[cohort_name]
        
        if split_column not in cohort_df.columns:
            logging.error(f"Column '{split_column}' not found in cohort '{cohort_name}'")
            return {}
        
        split_cohorts = {}
        available_values = cohort_df[split_column].dropna().unique()
        
        for value in values:
            if value in available_values:
                subset_mask = cohort_df[split_column] == value
                subset_df = cohort_df[subset_mask].copy()
                
                if len(subset_df) >= 20:  # Minimum for meaningful analysis
                    subset_name = f"{cohort_name}_{value.replace(' ', '_').lower()}"
                    subset_df.attrs.update({
                        'cohort_name': subset_name,
                        'parent_cohort': cohort_name,
                        'split_column': split_column,
                        'split_value': value,
                        'sample_size': len(subset_df)
                    })
                    split_cohorts[subset_name] = subset_df
                    logging.info(f"Created split cohort '{subset_name}' with {len(subset_df)} isolates")
        
        return split_cohorts
    
    def create_icu_ward_comparison(self) -> Dict[str, pd.DataFrame]:
        """Convenience method for ICU vs Ward comparison"""
        stewardship_cohorts = self._create_stewardship_cohorts()
        
        if 'stewardship_icu_vs_ward' in stewardship_cohorts:
            return self.split_cohort_by_metadata(
                cohort_name='stewardship_icu_vs_ward',
                split_column='ARS_WardType',
                values=['Intensive Care Unit', 'General Ward']
            )
        return {}

# =============================================================================
# 5. ARM STRATEGY
# =============================================================================

class ARMCohortStrategy(NetworkComparisonStrategy):
    """
    Prioritizes cohorts for Association Rule Mining based on data characteristics
    """
    
    def prioritize_cohorts_for_arm(self, use_case_cohorts: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[Dict]]:
        """Recommends which cohorts to prioritize for ARM analysis"""
        priority_recommendations = {}
        
        for use_case, cohorts in use_case_cohorts.items():
            priority_cohorts = []
            
            for cohort_name, cohort_df in cohorts.items():
                arm_suitability = self._assess_arm_suitability(cohort_df, cohort_name)
                
                if arm_suitability['priority_level'] in ['high', 'medium']:
                    priority_cohorts.append({
                        'cohort_name': cohort_name,
                        'priority_level': arm_suitability['priority_level'],
                        'reasoning': arm_suitability['reasoning'],
                        'expected_arm_insights': arm_suitability['expected_arm_insights']
                    })
            
            priority_cohorts.sort(key=lambda x: 0 if x['priority_level'] == 'high' else 1)
            priority_recommendations[use_case] = priority_cohorts
        
        return priority_recommendations
    
    def _assess_arm_suitability(self, cohort_df: pd.DataFrame, cohort_name: str) -> Dict[str, Any]:
        """Assess how suitable a cohort is for ARM analysis"""
        tested_columns = [col for col in cohort_df.columns if col.endswith('_Tested')]
        tested_data = cohort_df[tested_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        n_isolates = len(cohort_df)
        n_antibiotics = len(tested_columns)
        testing_density = tested_data.sum().sum() / (n_isolates * n_antibiotics)
        
        # High priority conditions
        if n_isolates < 50:
            return {
                'priority_level': 'low',
                'reasoning': f'Small sample size (n={n_isolates}) may yield unreliable rules',
                'expected_arm_insights': 'Limited - rules may not generalize'
            }
        
        high_priority_conditions = [
            ('cascade' in cohort_name and testing_density > 0.3, 
             "Cascade cohorts with high testing density should reveal strong 'if-then' rules"),
            (cohort_name.startswith('context_'),
             "Context comparison cohorts can reveal context-specific decision logic")
        ]
        
        for condition, reasoning in high_priority_conditions:
            if condition:
                return {
                    'priority_level': 'high',
                    'reasoning': reasoning,
                    'expected_arm_insights': self._get_expected_arm_insights(cohort_name),
                    'metrics': {
                        'n_isolates': n_isolates,
                        'n_antibiotics': n_antibiotics,
                        'testing_density': testing_density
                    }
                }
        
        return {
            'priority_level': 'medium',
            'reasoning': 'Moderate sample size and testing patterns may yield useful rules',
            'expected_arm_insights': self._get_expected_arm_insights(cohort_name),
            'metrics': {
                'n_isolates': n_isolates,
                'n_antibiotics': n_antibiotics,
                'testing_density': testing_density
            }
        }
    
    def _get_expected_arm_insights(self, cohort_name: str) -> List[str]:
        """Define what ARM is expected to reveal for each cohort type"""
        insights_map = {
            'cascade': [
                "High-confidence cascade rules (if antibiotic A → then antibiotic B)",
                "Identification of 'trigger' antibiotics that initiate broad testing"
            ],
            'context': [
                "Context-specific decision rules that would be masked in aggregate data",
                "Different 'if-then' logic for same pathogen in different clinical scenarios"
            ],
            'stewardship': [
                "Institutional testing protocols for Watch/Reserve antibiotics",
                "Rules revealing conservative vs. aggressive testing approaches"
            ],
            'evolution': [
                "Temporal evolution of testing decision rules",
                "Emergence or disappearance of specific testing associations"
            ]
        }
        
        for key, insights in insights_map.items():
            if key in cohort_name:
                return insights
        
        return ["General testing pattern rules and associations"]

# =============================================================================
# 6. COMPLETE ANALYSIS ORCHESTRATOR
# =============================================================================

class CohortAnalysisOrchestrator(ARMCohortStrategy):
    """
    Complete orchestrator that handles the entire analysis pipeline
    """
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline for all use cases"""
        
        # 1. Create all use case cohorts
        use_case_cohorts = self.create_use_case_cohorts()
        
        # 2. Get ARM priorities
        arm_priorities = self.prioritize_cohorts_for_arm(use_case_cohorts)
        
        # 3. Create ICU vs Ward comparison
        icu_ward_comparison = self.create_icu_ward_comparison()
        
        results = {
            'use_case_cohorts': use_case_cohorts,
            'arm_priorities': arm_priorities,
            'icu_ward_comparison': icu_ward_comparison,
            'summary': self._generate_summary(use_case_cohorts)
        }
        
        logging.info("Complete analysis pipeline finished successfully")
        return results
    
    def _generate_summary(self, use_case_cohorts: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Generate summary statistics for the analysis"""
        summary = {
            'total_cohorts_created': 0,
            'total_isolates': 0,
            'cohorts_by_use_case': {},
            'sample_size_range': {}
        }
        
        for use_case, cohorts in use_case_cohorts.items():
            summary['cohorts_by_use_case'][use_case] = len(cohorts)
            summary['total_cohorts_created'] += len(cohorts)
            
            sizes = [len(cohort_df) for cohort_df in cohorts.values()]
            if sizes:
                summary['total_isolates'] += sum(sizes)
                summary['sample_size_range'][use_case] = {
                    'min': min(sizes),
                    'max': max(sizes),
                    'mean': int(np.mean(sizes))
                }
        
        return summary



def load_dataset(df:pd.DataFrame):
    """Example usage of the complete system"""
    # Initialize the orchestrator
    orchestrator = CohortAnalysisOrchestrator(df)
    
    # Run complete analysis
    results = orchestrator.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    summary = results['summary']
    print(f"Total cohorts created: {summary['total_cohorts_created']}")
    print(f"Total isolates analyzed: {summary['total_isolates']}")
    
    for use_case, count in summary['cohorts_by_use_case'].items():
        print(f"  {use_case}: {count} cohorts")
    
    # ARM priorities
    print("\nARM ANALYSIS PRIORITIES:")
    for use_case, priorities in results['arm_priorities'].items():
        print(f"\n{use_case}:")
        for priority in priorities:
            print(f"  {priority['cohort_name']} - {priority['priority_level']}: {priority['reasoning']}")
    
    # ICU vs Ward comparison
    if results['icu_ward_comparison']:
        print(f"\nICU vs WARD COMPARISON:")
        for cohort_name, cohort_data in results['icu_ward_comparison'].items():
            print(f"  {cohort_name}: {len(cohort_data)} isolates")
