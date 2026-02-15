import pandas as pd
from mappers.state_city_mapper import state_mapping
class StateAssigner:
    def __init__(self):
        self.state_mapping = state_mapping
    
    def assign_state(self, kreis_name):
        for state, cities in self.state_mapping.items():
            if kreis_name in cities:
                return state
        return "Unknown"

    def update_dataframe(self, file_path, output_path):
        df = pd.read_csv(file_path)
        df["state"] = df["kreis_name"].apply(self.assign_state)
        
        # Group by state and year and select the mean of gisd_score
        grouped_data = df.groupby(["state", "year"], as_index=False)[["gisd_score"]].mean()
        grouped_data["year"] = grouped_data["year"].astype(int)  # Convert year back to integer

        # Select records from 2019 to 2023
        filtered_data = grouped_data[(grouped_data['year'] >= 2019) & (grouped_data['year'] <= 2023)]

        filtered_data.to_csv("./datasets/GISD_Bundesland_Updated.csv", index=False)

        df.to_csv(output_path, index=False)
        print("State column updated successfully!")

if __name__ == "__main__":
    state_assigner = StateAssigner()
    state_assigner.update_dataframe(
        "./datasets/GISD_/GISD_Bundesland_CSVs/GISD_Bund_Kreis.csv", 
        "./datasets/GISD_Bund_Kreis_Updated.csv"
    )

