import pandas as pd

class PopulationDataFormatter:
    def __init__(self, file_path, output_file):
        self.file_path = file_path
        self.output_file = output_file
        self.df = None

    def load_data(self):
        """Loads the CSV file with proper encoding and separator."""
        self.df = pd.read_csv(self.file_path, encoding="latin1", sep=";", header=None)

    def extract_headers(self):
        """Extracts column headers from the first two rows."""
        years = self.df.iloc[0, 1:].values.tolist()
        genders = self.df.iloc[1, 1:].values.tolist()
        self.df = self.df[2:].reset_index(drop=True)
        columns = ["bundesland"] + [f"{year}_{gender}" for year, gender in zip(years, genders)]
        self.df.columns = columns

    def melt_data(self):
        """Converts data into long format."""
        self.df = self.df.melt(id_vars=["bundesland"], var_name="Year_Gender", value_name="Population")

    def extract_year_gender(self):
        """Extracts year and gender from the merged column."""
        self.df["Year"] = self.df["Year_Gender"].str.extract(r"(\d{4})").astype(int)
        self.df["Gender"] = self.df["Year_Gender"].str.extract(r"(Insgesamt|mï¿½nnlich|weiblich)")
        
        # Map gender names to English equivalents
        gender_map = {"Insgesamt": "total", "mï¿½nnlich": "male", "weiblich": "female"}
        self.df["Gender"] = self.df["Gender"].map(gender_map)

    def pivot_data(self):
        """Pivots the table to match the required format."""
        df_pivoted = self.df.pivot(index=["bundesland", "Year"], 
                                   columns="Gender", 
                                   values="Population").reset_index()
        
        # Ensure all gender columns exist
        for col in ["male", "female", "total"]:
            if col not in df_pivoted.columns:
                df_pivoted[col] = None
        
        # Convert Population column to numeric to avoid data type issues
        df_pivoted["male"] = pd.to_numeric(df_pivoted["male"], errors='coerce')
        df_pivoted["female"] = pd.to_numeric(df_pivoted["female"], errors='coerce')
        df_pivoted["total"] = pd.to_numeric(df_pivoted["total"], errors='coerce')

        self.df = df_pivoted

    def rename_columns_and_clean(self):
        """Renames columns and standardizes 'bundesland' values."""
        self.df = self.df.rename(columns={"total": "total", "male": "male_count", "female": "female_count"})
        self.df = self.df[["bundesland", "Year", "male_count", "female_count", "total"]]

        # Fix known encoding issues in 'bundesland' names
        self.df["bundesland"] = self.df["bundesland"].replace({"Baden-Wï¿½rttemberg": "Baden-Württemberg"})

    def process(self):
        """Executes all steps to format the population data."""
        self.load_data()
        self.extract_headers()
        self.melt_data()
        self.extract_year_gender()
        self.pivot_data()
        self.rename_columns_and_clean()
        self.df.to_csv(self.output_file, index=False)
        return self.df

# Example Usage
file_path = "./datasets/other_data/Population-data-gender-2019-2023-12411-01-01-5-B.csv"
output_file = "./datasets/population-data-cleaned.csv"
formatter = PopulationDataFormatter(file_path, output_file)
df_cleaned = formatter.process()

# Display the cleaned dataframe
print(df_cleaned.head())

