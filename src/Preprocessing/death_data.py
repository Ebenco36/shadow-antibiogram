import pandas as pd
import unicodedata

class DeathRateDataProcessor:
    def __init__(self, file_paths, output_file):
        """
        Initialize the processor with a list of file paths.
        
        :param file_paths: List of CSV file paths to be processed.
        """
        self.file_paths = file_paths
        self.output_file = output_file
        self.df_combined = None

    def load_data(self, file_path):
        """
        Loads a CSV file into a pandas DataFrame.
        
        :param file_path: Path to the CSV file.
        :return: DataFrame containing the loaded data.
        """
        return pd.read_csv(file_path, encoding="latin1")

    def process_death_rate_data(self, df):
        """
        Transforms the wide-format death rate data into a long format.
        
        :param df: Input DataFrame in wide format.
        :return: Transformed DataFrame in long format.
        """
        # Extract headers for day-month values
        day_months = df.iloc[0, 2:-1].values.tolist()  # Skip 'Jahr' and 'Bundesland', exclude 'Insgesamt'

        # Extract actual data, dropping the first row of headers
        df = df[1:].reset_index(drop=True)
        df.columns = ["Year", "Bundesland"] + day_months + ["Total"]

        # Convert year to integer
        df["Year"] = pd.to_numeric(df["Year"], errors='coerce').astype("Int64")

        # Convert to long format
        df_melted = df.melt(id_vars=["Year", "Bundesland"], var_name="Day-Month", value_name="Value")

        # Remove the 'Total' column rows
        df_melted = df_melted[df_melted["Day-Month"] != "Total"].reset_index(drop=True)

        # Convert Value column to numeric
        df_melted["Value"] = pd.to_numeric(df_melted["Value"], errors="coerce")

        # Correct the date format
        df_melted["Date"] =  df_melted["Day-Month"] + df_melted["Year"].astype(str)

        # Fix encoding issues in 'Bundesland' names
        df_melted["Bundesland"] = df_melted["Bundesland"].apply(self.fix_encoding)

        return df_melted

    def fix_encoding(self, text):
        """
        Corrects encoding issues in Bundesland names using a predefined mapping.
        
        :param text: Input string with potential encoding issues.
        :return: Corrected string.
        """
        if pd.isna(text):
            return text

        mapping = {
            "Brandenburg": "Brandenburg",
            "Berlin": "Berlin", 
            "Bremen": "Bremen", 
            "Hessen": "Hessen", 
            "Hamburg": "Hamburg",
            "Nordrhein-Westfalen": "Nordrhein-Westfalen",
            "Sachsen": "Sachsen", 
            "Bayern": "Bayern", 
            "Niedersachsen": "Niedersachsen",
            "Rheinland-Pfalz": "Rheinland-Pfalz",
            "Saarland": "Saarland",
            "Sachsen-Anhalt": "Sachsen-Anhalt",
            "ThÃ¼ringen": "Thüringen",
            "Schleswig-Holstein": "Schleswig-Holstein",
            "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
            "Baden-WÃ¼rttemberg": "Baden-Württemberg"
        }
        
        return mapping.get(text, text)

    def process_files(self):
        """
        Processes all files, combines the data, and stores it in self.df_combined.
        
        :return: Combined and processed DataFrame.
        """
        processed_data = []

        for file_path in self.file_paths:
            df = self.load_data(file_path)
            processed_df = self.process_death_rate_data(df)
            processed_data.append(processed_df)

        # Combine all processed data
        self.df_combined = pd.concat(processed_data, ignore_index=True)

        # Filter for specific years
        self.df_combined = self.df_combined[self.df_combined["Year"].isin([2019, 2020, 2021, 2022, 2023])]
        self.df_combined.to_csv(self.output_file, index=False)
        return self.df_combined

# Example Usage
file_paths = [
    "./datasets/other_data/Death-rate-data-by-day-5126108209005_SB.xlsx - 12613-08.csv",
    "./datasets/other_data/Death-rate-data-per-day-from-2021-2025-statistischer-bericht-sterbefaelle-tage-wochen-monate-aktuell-5126109 (1).xlsx - 12613-08.csv"
]
output_file = "./datasets/death_rate_data_combined.csv"
processor = DeathRateDataProcessor(file_paths, output_file)
df_combined = processor.process_files()