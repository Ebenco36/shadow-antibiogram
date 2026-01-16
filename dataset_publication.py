from src.controllers.DataLoader import DataLoader
from src.runners.DataProcessing import save_parquet_flat

loader = DataLoader("./datasets/WHO_Aware_data____")
exempt_cols = ['NumberOrganisation', 'OrgType', 'Anonymized_Lab']
df = loader.get_combined()
features_df = df.drop(columns=exempt_cols, errors='ignore')
save_parquet_flat(
    df=features_df,
    out_dir="./datasets/WHO_Aware_data",
    rows_per_file=250_000
)