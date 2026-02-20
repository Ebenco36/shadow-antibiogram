import pandas as pd
import os

class DataPreprocessor:
    def __init__(self, data):
        """
        Initialize the DataPreprocessor class.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
        """
        self.data = data

    def set_column_value(self, column, new_value, condition):
        """
        Set values in a column based on a given condition.

        Parameters:
            column (str): The column to modify.
            new_value: The new value to assign.
            condition (callable or dict):
                - If a function, it should return a boolean Series.
                - If a dict, it should map column names to values to check for equality.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        if callable(condition):
            self.data.loc[condition(self.data), column] = new_value
        elif isinstance(condition, dict):
            mask = pd.Series(True, index=self.data.index)
            for col, val in condition.items():
                mask &= self.data[col] == val
            self.data.loc[mask, column] = new_value
        else:
            raise ValueError("Condition must be a function or a dictionary.")

        return self.data

    def replace_values(self, column, replacements):
        """
        Replace multiple values in a column using a dictionary.

        Parameters:
            column (str): The column to update.
            replacements (dict): Dictionary mapping old values to new values.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        if column in self.data.columns:
            self.data[column] = self.data[column].replace(replacements)
        else:
            raise KeyError(f"Column '{column}' not found in data.")
        
        return self.data
    
    
    def rename_columns_from_mapping(self, mapping_df, key_col, value_col):
        """
        Rename columns in the DataFrame based on a mapping DataFrame.

        Parameters:
            mapping_df (pd.DataFrame): The DataFrame containing old and new column names.
            key_col (str): The column in mapping_df containing the old column names.
            value_col (str): The column in mapping_df containing the new column names.

        Returns:
            pd.DataFrame: The modified DataFrame with renamed columns.
        """
        rename_dict = dict(zip(mapping_df[key_col], mapping_df[value_col]))
        self.data.rename(columns=rename_dict, inplace=True)
        return self.data

    def save_data(self, file_path):
        """
        Save the processed DataFrame to a CSV file.

        Parameters:
            file_path (str): The path where the CSV file will be saved.

        Returns:
            None
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
        self.data.to_csv(file_path, index=False)
        print(f"Data saved successfully at: {file_path}")

