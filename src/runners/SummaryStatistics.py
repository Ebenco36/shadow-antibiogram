import pandas as pd
from src.controllers.AMRTestingBasicStatistics import AMRTestingPatternAnalysis


def run_visualizations(main_df:pd.DataFrame):
    analyzer = AMRTestingPatternAnalysis(
        output_path="./outputs",
        dataframe=main_df,
    )

    extra_dir_to_save = "summary_statistics/"
    research_questions = [
        # --- 1. Overall Epidemiology & Surveillance Scope ---
        {
            "question": "Top 15 Pathogen Genus (2019 - 2023)",
            "params": {
                "field": "PathogenGenus",
                "top_n": 15,
                "orientation": "horizontal",
                "height": 600,
                "file_name": "01_top_15_pathogens_genus",
                "color": "#3182bd"
            }
        },
        {
            "question": "Top 15 Pathogens (2019 - 2023)",
            "params": {
                "field": "Pathogen",
                "top_n": 15,
                "orientation": "horizontal",
                "height": 600,
                "file_name": "01_top_15_pathogens",
                "color": "#3182bd"
            }
        },
        {
            "question": "Distribution of Sample Materials (2019 - 2023)",
            "params": {
                "field": "TextMaterialgroupRkiL0",
                "top_n": 15,
                "orientation": "horizontal",
                "height": 600,
                "file_name": "02_sample_material_distribution",
                "color": "#3182bd"
            }
        },
        {
            "question": "Patient Age Range Distribution Across All Isolates (2019 - 2023)",
            "params": {
                "field": "AgeRange",
                "custom_sort_order": [
                    '0 years', '1-4 years', '5-9 years', '10-14 years', '15-19 years',
                    '20-24 years', '25-29 years', '30-34 years', '35-39 years',
                    '40-44 years', '45-49 years', '50-54 years', '55-59 years',
                    '60-64 years', '65-69 years', '70-74 years', '75-79 years',
                    '80-84 years', '85-89 years', '90-94 years', '≥95 years'
                ],
                "file_name": "03_patient_age_distribution",
                "color": "#3182bd",
                "width": 1200,
                "height": 600,
                "sort_order": "ascending",
                "orientation": "vertical",
            }
        },
        
        {
            "question": "Patient Age Group Distribution Across All Isolates (2019 - 2023)",
            "params": {
                "field": "AgeGroup",
                "custom_sort_order": [
                    'Pediatric',
                    'Adolescents & Young Adults',
                    'Adult', 'Elderly',
                ],
                "file_name": "03_patient_age_group_distribution",
                "color": "#3182bd",
                "sort_order": "ascending",
                "orientation": "horizontal",
            }
        },
        

        # --- 2. Investigating Disparities & Stratification ---
        {
            "question": "Top 12 Pathogens by Care Type (Inpatient vs. Outpatient)",
            "params": {
                "field": "Pathogen",
                "top_n": 15,
                "stack_by": "CareType",
                "stack_colors": ["#9ecae1", "#3182bd"],
                "height": 600,
                "file_name": "05_pathogens_by_caretype",
                # "color": "#3182bd"
            }
        },
        {
            "question": "Q5: For the top pathogens, what is their prevalence by patient sex?",
            "params": {
                "field": "Pathogen",
                "top_n": 20,
                "stack_by": "Sex",
                "stack_colors": ["#cccccc", "#969696", "#525252"],
                "height": 600,
                "file_name": "06_pathogens_by_sex",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q6: How has the distribution of the top 5 pathogens changed over the years?",
            "params": {
                "field": "Year",
                "stack_by": "Pathogen",
                "stack_colors": [
                    "#f0f9e8",
                    "#bae4bc",
                    "#7bccc4",
                    "#43a2ca",
                    "#0868ac"
                ],
                "prefilter": lambda data: data[data['Pathogen'].isin(data['Pathogen'].value_counts().nlargest(5).index)],
                "file_name": "07_top_5_pathogens_by_year",
                "color": "#3182bd"
            }
        },

        # --- 3. Targeted, Hypothesis-Driven Scenarios ---
        {
            "question": "Q7: What are the most common pathogens found specifically in Blood Cultures?",
            "params": {
                "field": "Pathogen",
                "top_n": 10,
                "orientation": "horizontal",
                "prefilter": lambda data: data[data['TextMaterialgroupRkiL0'] == 'Blood Culture'],
                "file_name": "08_top_pathogens_in_blood_cultures",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q8: Among isolates from Intensive Care Units (ICUs), what are the most frequent pathogens?",
            "params": {
                "field": "Pathogen",
                "top_n": 10,
                "orientation": "horizontal",
                "prefilter": lambda data: data[data['ARS_WardType'] == 'Intensive Care Unit'],
                "file_name": "09_top_pathogens_in_icu",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q9: For Escherichia coli infections, what is the patient age distribution?",
            "params": {
                "field": "AgeGroup",
                "custom_sort_order": ['Pediatric', 'Adolescents & Young Adults', 'Adult', 'Elderly'],
                "prefilter": lambda data: data[data['Pathogen'] == 'Escherichia coli'],
                "file_name": "10_age_distribution_for_ecoli",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q9A: For Escherichia coli infections, what is the patient age range distribution?",
            "params": {
                "field": "AgeRange",
                "custom_sort_order": [
                    '0 years', '1-4 years', '5-9 years', '10-14 years', '15-19 years',
                    '20-24 years', '25-29 years', '30-34 years', '35-39 years',
                    '40-44 years', '45-49 years', '50-54 years', '55-59 years',
                    '60-64 years', '65-69 years', '70-74 years', '75-79 years',
                    '80-84 years', '85-89 years', '90-94 years', '≥95 years'
                ],
                "prefilter": lambda data: data[data['Pathogen'] == 'Escherichia coli'],
                "file_name": "10_age_range_distribution_for_ecoli",
                "color": "#3182bd",
                "sort_order": "ascending",
            }
        },
        # --- 4. Geographic and System-Level Disparities ---
        {
            "question": "Q10: What is the distribution of isolates across German states (Bundesländer)?",
            "params": {
                "field": "Bundesland",
                "top_n": 16,
                "orientation": "horizontal",
                "height": 600,
                "file_name": "11_distribution_by_bundesland",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q11: How does the pathogen profile differ between major German regions (ARS_Region)?",
            "params": {
                "field": "ARS_Region",
                "stack_by": "Pathogen",
                "stack_colors": [
                    "#f0f9e8",
                    "#bae4bc",
                    "#7bccc4",
                    "#43a2ca",
                    "#0868ac"
                ],
                "prefilter": lambda data: data[data['Pathogen'].isin(data['Pathogen'].value_counts().nlargest(5).index)],
                "file_name": "12_pathogen_profiles_by_region",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q12: What is the distribution of samples across different hospital care levels?",
            "params": {
                "field": "ARS_HospitalLevelManual",
                "orientation": "horizontal",
                "height": 600,
                "file_name": "13_distribution_by_hospital_level",
                "custom_sort_order": [
                    "Level 1 - Basic Care",
                    "Level 2 - Regular Care",
                    "Level 3 - Specialized Care",
                    "Level 4 - Maximum Care",
                    "Level 5 - Specialized Hospitals",
                    "Level 7 - Preventive and Rehabilitation Facilities",
                    "Level 6 - Other Hospitals",
                    "Not Assigned"
                ],
                "sort_order": "ascending",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q13: What is the surveillance contribution of the top 20 laboratories?",
            "params": {
                "field": "ARS_LabName",
                "top_n": 20,
                "orientation": "horizontal",
                "height": 600,
                "file_name": "14_contribution_by_lab",
                "color": "#3182bd"
            }
        },

        # --- 5. Clinical Setting and Microbiological Variations ---
        {
            "question": "Q14: What is the distribution of Gram-positive vs. Gram-negative isolates?",
            "params": {
                "field": "GramType",
                "file_name": "15_distribution_by_gramtype",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q15: How does the Gram-type distribution change between ICUs and Normal Wards?",
            "params": {
                "field": "ARS_WardType",
                "stack_by": "GramType",
                "stack_colors": [
                    "#a6611a",
                    "#dfc27d",
                    "#80cdc1",
                    "#018571"
                ],
                "prefilter": lambda data: data[data['ARS_WardType'].isin(['Intensive Care Unit', 'Normal Ward'])],
                "file_name": "16_gramtype_by_ward",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q16: What is the breakdown of inpatient vs. outpatient care across the dataset?",
            "params": {
                "field": "CareType",
                "file_name": "17_caretype_distribution",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q17: For blood culture isolates, what is the sex distribution of patients?",
            "params": {
                "field": "Sex",
                "prefilter": lambda data: data[data['TextMaterialgroupRkiL0'] == 'Blood Culture'],
                "file_name": "18_sex_distribution_in_blood_cultures",
                "color": "#3182bd"
            }
        },
        {
            "question": "Q19: What is the overall seasonal distribution of all collected isolates?",
            "params": {
                "field": "SeasonName",
                "color": "#3182bd",
                # Custom sort order is critical for seasons to appear in logical order
                "custom_sort_order": ['Spring', 'Summer', 'Fall', 'Winter'],
                "file_name": "19_overall_seasonal_distribution"
            }
        },
        {
            "question": "Q20: How does the prevalence of the top 6 pathogens vary by season?",
            "params": {
                "field": "SeasonName",
                "stack_by": "Pathogen",
                "custom_sort_order": ['Spring', 'Summer', 'Fall', 'Winter'],
                # Prefilter to keep the chart clean and focused on major pathogens
                "prefilter": lambda data: data[data['Pathogen'].isin(data['Pathogen'].value_counts().nlargest(6).index)],
                "height": 500,
                "file_name": "20_top_6_pathogens_by_season"
            }
        },
        {
            "question": "Q21: Is there a distinct seasonal pattern for isolates collected from outpatient ('ambulant') settings?",
            "params": {
                "field": "SeasonName",
                "custom_sort_order": ['Spring', 'Summer', 'Fall', 'Winter'],
                "prefilter": lambda data: data[data['CareType'] == 'ambulant'],
                "file_name": "21_seasonal_distribution_outpatient"
            }
        },
        {
            "question": "Q22: What are the top 10 pathogens for inpatients versus outpatients?",
            "params": {
                "field": "Pathogen",
                "stack_by": "CareType",
                "top_n": 10,
                "orientation": "horizontal",
                "height": 500,
                "file_name": "22_top_pathogens_by_caretype"
            }
        },
        {
            "question": "Q23: How does the distribution of sample material types differ between care settings?",
            "params": {
                "field": "TextMaterialgroupRkiL0",
                "stack_by": "CareType",
                "top_n": 10,
                "orientation": "horizontal",
                "height": 500,
                "file_name": "23_sample_type_by_caretype"
            }
        },
        {
            "question": "Q24: What is the patient age distribution for inpatients versus outpatients?",
            "params": {
                "field": "AgeRange",
                "stack_by": "CareType",
                "custom_sort_order": [
                    '0 years', '1-4 years', '5-9 years', '10-14 years', '15-19 years',
                    '20-24 years', '25-29 years', '30-34 years', '35-39 years',
                    '40-44 years', '45-49 years', '50-54 years', '55-59 years',
                    '60-64 years', '65-69 years', '70-74 years', '75-79 years',
                    '80-84 years', '85-89 years', '90-94 years', '≥95 years'
                ],
                "file_name": "24_age_distribution_by_caretype",
                "sort_order": "ascending",
            }
        },
        {
            "question": "Q25: What is the patient age range distribution",
            "params": {
                "field": "AgeRange",
                "file_name": "25_age_range_distribution"
            }
        }
    ]
    for item in research_questions:
        question = item["question"]
        params = item["params"]
        
        print(f"Analyzing... {question}")
        
        try:
            # Pass the 'question' string directly to the new 'title' parameter
            result_df = analyzer.top_values_with_tests(
                **params, 
                chart_title_param=question,
                extra_dir_to_save=extra_dir_to_save
            )
            
            if not result_df.empty:
                print(f"Successfully generated chart and table for: {params.get('file_name')}")
            else:
                print(f"Warning: No data returned for '{question}'.")
            print("-" * 50)

        except Exception as e:
            print(f"[ERROR] Failed to answer '{question}'. Reason: {e}")
            print("-" * 50)
  
    #################################### STOPPED HERE FOR NOW #########################################
from src.controllers.DataLoader import DataLoader

if __name__ == "__main__":
    #################################################################################
    ##################### LOAD DATASETS FOR ALL PROCESSES BELOW #####################
    #################################################################################

    loader = DataLoader("./datasets/WHO_Aware_data")
    df_combined = loader.get_combined()
    df = df_combined
    run_visualizations(main_df=df)