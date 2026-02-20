import pandas as pd
import altair as alt

alt.data_transformers.enable('default')

class LoadClasses:
    def __init__(
        self,
        antibiotics_class_file = "./datasets/antibiotic_classification_complete.csv",
        suffix = "_Tested"
    ):
        self.suffix = suffix
        self.antibiotics_class_file = antibiotics_class_file
        self.antibiotic_class_data = self.load_antibiotic_classification(antibiotic_classification_file=antibiotics_class_file)
        self.antibiotic_class_list = self.antibiotic_class_data["Class"].unique().tolist()
    
    def set_suffix(self, suffix = "_Tested"):
        self.suffix = suffix
        return self
        
    def convert_to_tested_columns(self, antibiotics_list):
        if not isinstance(antibiotics_list, list):
            raise ValueError("Input must be a list of antibiotic names.")
        
        return [f"{antibiotic}{self.suffix}" for antibiotic in antibiotics_list]

    def load_antibiotic_classification(
        self, 
        antibiotic_classification_file="/scratch/projekte/FG37_ARS/AMR_code/datasets/antibiotic_classification_complete.csv"
    ):
        return pd.read_csv(antibiotic_classification_file)

    def get_antibiotics_by_class(self, antibiotic_classes):
        if isinstance(antibiotic_classes, str):
            antibiotic_classes = [antibiotic_classes]  # Convert single class to list

        if not isinstance(antibiotic_classes, list):
            raise ValueError("Antibiotic classes must be a string or a list of strings.")

        # Ensure all provided classes exist
        
        invalid_classes = [cls for cls in antibiotic_classes if cls not in self.antibiotic_class_list]
        
        if invalid_classes:
            raise ValueError(f"Invalid class(es) found: {', '.join(invalid_classes)}")

        # Retrieve antibiotics from all selected classes
        selected_antibiotics = self.antibiotic_class_data[
            self.antibiotic_class_data["Class"].isin(antibiotic_classes)
        ]["Full Name"].tolist()

        return list(set(selected_antibiotics))  # Remove duplicates

    def get_antibiotics_by_category(self, categories):
        if isinstance(categories, str):
            categories = [categories]  # Convert single class to list

        if not isinstance(categories, list):
            raise ValueError("categories must be a string or a list of strings.")
        
        # Ensure all provided classes exist
        valid_categories = self.antibiotic_class_data["Category"].unique().tolist()
        invalid_categories = [cls for cls in categories if cls not in valid_categories]
        
        if invalid_categories:
            raise ValueError(f"Invalid class(es) found: {', '.join(invalid_categories)}")

        # Retrieve antibiotics from all selected classes
        selected_antibiotics = self.antibiotic_class_data[
            self.antibiotic_class_data["Category"].isin(categories)
        ]["Full Name"].tolist()

        return list(set(selected_antibiotics))  # Remove duplicates
    
    
    
    def get_antibiotics_by_broad_class(self, broad_classes, broad_col: str = "Broad Class"):
        """
        Return a list of antibiotic 'Full Name's belonging to one or more broad pharmacology classes.

        Flexible behavior:
        - If some requested broad classes are not present in the data, they are ignored.
        - If none of the requested classes are present, an empty list is returned.

        Parameters
        ----------
        broad_classes : str or list of str
            Broad class names, e.g. "β-lactam", "Aminoglycoside", "Fluoroquinolone".
        broad_col : str
            Column name in the classification CSV that contains broad class labels.

        Returns
        -------
        list of str
            Unique antibiotic full names for the requested *valid* broad class(es).
            If no valid classes are found, returns an empty list.
        """

        # Ensure the broad class column exists
        if broad_col not in self.antibiotic_class_data.columns:
            raise ValueError(
                f"Column '{broad_col}' not found in antibiotic_class_data. "
                f"Make sure your merged classification file includes a '{broad_col}' column."
            )

        # Normalise input to list
        if isinstance(broad_classes, str):
            broad_classes = [broad_classes]

        if not isinstance(broad_classes, list):
            raise ValueError("broad_classes must be a string or a list of strings.")

        # All distinct values present in the data
        valid_broad_classes = (
            self.antibiotic_class_data[broad_col]
            .dropna()
            .unique()
            .tolist()
        )

        # Split into valid and invalid
        requested = list(dict.fromkeys(broad_classes))  # dedupe, preserve order
        valid_requested = [cls for cls in requested if cls in valid_broad_classes]
        invalid_requested = [cls for cls in requested if cls not in valid_broad_classes]

        # Soft behavior: warn but do NOT raise if some are invalid
        if invalid_requested:
            print(
                f"[LoadClasses.get_antibiotics_by_broad_class] "
                f"Ignoring unknown broad class(es): {', '.join(map(str, invalid_requested))}. "
                f"Valid options include: {', '.join(sorted(map(str, valid_broad_classes)))}"
            )

        # If none of the requested classes exist, just return empty list
        if not valid_requested:
            return []

        # Select antibiotics that belong to any of the *valid* requested classes
        mask = self.antibiotic_class_data[broad_col].isin(valid_requested)
        selected_antibiotics = self.antibiotic_class_data.loc[mask, "Full Name"].tolist()

        # Deduplicate
        return list(set(selected_antibiotics))



    